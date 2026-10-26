"""Fixed prediction-only spatial aggregation and mask correction."""

import numpy as np
from scipy import ndimage

SCORE_NAMES = ('image_success_max', 'image_success_area_mean', 'patch_success_area_mean', 'patch_success_max', 'patch_seed_area_fraction', 'patch_seed_slice_fraction', 'seeded_slice_count', 'slice_span', 'joint_geometric_support')
GEOMETRY_NAMES = ('log_local_area', 'log_local_patches', 'local_y', 'local_x', 'local_height', 'local_width', 'local_occupancy', 'local_fill', 'log_component_area', 'log_component_patches', 'log_component_slices', 'log_component_span', 'component_slice_coverage', 'component_height', 'component_width', 'local_area_fraction', 'local_patch_fraction', 'relative_z', 'centroid_displacement', 'previous_area_ratio', 'next_area_ratio')
SPEC_KEYS = ('checkpoint', 'concept_count_policy', 'patch_failure_cutoff', 'image_failure_cutoff', 'region_success_cutoff')


def boundary_features(area, raw_area):
    area, raw_area = np.asarray(area), np.asarray(raw_area)
    if (area.shape != (len(raw_area), 3) or (raw_area <= 0).any() or
            (area < 0).any() or (area > raw_area[:, None]).any() or
            (np.diff(area, axis=1) > 0).any()):
        raise ValueError('Nested predicted areas within positive raw regions required')
    return np.column_stack((np.log1p(area), area/raw_area[:, None],
        (area[:, :2]-area[:, 2:])/raw_area[:, None])).astype(np.float32)

def choose_radii(scores, margin):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all() or not 0 <= margin <= 1:
        raise ValueError('Finite gain scores and valid margin required')
    # Prefer milder change (radius one) on a tie; reference wins at the margin.
    return np.where(np.maximum(scores[:, 0], scores[:, 1]) > margin,
        np.where(scores[:, 0] > scores[:, 1], 0, 1), 2)

def predict_gains(bundle, features, sample_rows, context_rows):
    ids = np.flatnonzero(np.isin(sample_rows, context_rows))
    scores = np.full((len(sample_rows), 2), np.nan)
    if len(ids):
        design = features[np.ix_(ids, bundle['variable_features'])]
        scores[ids] = np.column_stack([head.predict(design) for head in bundle['heads']]).clip(-1, 1)
        if not np.isfinite(scores[ids]).all():
            raise ValueError('Finite predicted Dice gains required')
    return scores

def preserve_empty_regions(radii, sample_keep, area):
    """Only already retained, nonempty regions may receive lighter erosion."""
    result = np.asarray(radii).copy()
    result[~(sample_keep & (area[:, 2] > 0))] = 2
    return result

def corrected_mask(prediction, old_mask, inputs, row, keep, groups, radii):
    """Add only the chosen lighter boundary; never remove an original pixel."""
    raw = mask_from_patches(prediction, inputs, row, keep)
    box = crop_bounds(raw, 3)
    if box is None:
        if old_mask.any():
            raise ValueError('Nonempty original mask has no raw foreground')
        return old_mask.copy()
    old_rebuilt = np.zeros_like(raw)
    old_rebuilt[box] = refine_mask(raw[box], minimum_component_pixels=256, boundary_offset_pixels=-2)
    np.testing.assert_array_equal(old_rebuilt, old_mask)
    ids = np.flatnonzero(inputs['patch_slice'] == row)
    grid = np.full((64, 64), 2, int)
    y, x = inputs['coords'][ids].T
    grid[y, x] = radii[groups['patch_sample'][ids]]
    local_radius = grid.repeat(16, 0).repeat(16, 1)[box]
    result = old_mask.copy()
    for radius in (0, 1):
        if np.any(local_radius == radius):
            result[box] |= refine_mask(raw[box], minimum_component_pixels=256,
                boundary_offset_pixels=-radius) & (local_radius == radius)
    if not np.array_equal(result.any(), old_mask.any()):
        raise ValueError('Boundary refinement must preserve original slice decisions')
    return result

def mask_from_patches(prediction, inputs, row, kept):
    ids = np.flatnonzero(inputs['patch_slice'] == row)
    grid = np.zeros((64, 64), bool)
    if len(ids):
        y, x = inputs['coords'][ids].T
        grid[y, x] = kept[ids]
    return prediction & grid.repeat(16, 0).repeat(16, 1)

def crop_bounds(mask, padding):
    """A box containing every output pixel for offsets <= padding."""
    y, x = np.nonzero(mask)
    if not len(y):
        return None
    return (slice(max(0, int(y.min())-padding), min(mask.shape[0], int(y.max())+padding+1)),
            slice(max(0, int(x.min())-padding), min(mask.shape[1], int(x.max())+padding+1)))

def refine_mask(mask, minimum_component_pixels=0, boundary_offset_pixels=0):
    """Refine a binary 2D mask without labels, including at image edges.

    The offset is a Euclidean radius in input pixels, not a physical distance.
    No foreground is created on an empty slice. Connected fragments are removed
    before the offset, so dilation cannot preserve otherwise rejected fragments.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Expected a binary 2D mask')
    if (not isinstance(minimum_component_pixels, (int, np.integer)) or
            minimum_component_pixels < 0 or
            not isinstance(boundary_offset_pixels, (int, np.integer))):
        raise ValueError('Area and signed offset must be integers; area must be nonnegative')
    result = mask.astype(bool, copy=True)
    if not result.any():
        return result
    if minimum_component_pixels:
        labels, _ = ndimage.label(result, structure=np.ones((3, 3), bool))
        keep = np.bincount(labels.ravel()) >= minimum_component_pixels
        keep[0] = False
        result = keep[labels]
    if not result.any() or boundary_offset_pixels == 0:
        return result
    if boundary_offset_pixels > 0:
        return ndimage.distance_transform_edt(~result) <= boundary_offset_pixels
    # Explicit exterior zeros avoid scipy's implicit edge convention when the
    # foreground touches the image edge or fills the complete image.
    distance = ndimage.distance_transform_edt(np.pad(result, 1))[1:-1, 1:-1]
    return distance > -boundary_offset_pixels

def retained_samples(keep, groups):
    result = np.zeros(len(groups['sample_slice']), bool)
    for sample, members in enumerate(groups['members']):
        if len(np.unique(keep[members])) != 1:
            raise ValueError('Keep or reject all patches in each region slice')
        result[sample] = keep[members[0]]
    return result

def rescue_empty_slices(reference, proposal, clean_patch_area, patch_slice, n_slices):
    """Select one complete mask per slice using reference predicted area only.

    The inputs are keep decisions for whole connected regions and patch areas
    after the same fixed cleanup. A nonempty reference is never modified. This
    rule must be applied after defining cleanup, since tiny raw fragments can
    disappear during cleanup. No target area or overlap is accepted here.
    """
    reference, proposal = np.asarray(reference), np.asarray(proposal)
    clean_patch_area, patch_slice = np.asarray(clean_patch_area), np.asarray(patch_slice)
    if (reference.dtype != bool or proposal.dtype != bool or reference.ndim != 1 or
            any(a.shape != reference.shape for a in (proposal, clean_patch_area, patch_slice))):
        raise ValueError('Aligned boolean decisions and one-dimensional patch arrays required')
    if (not isinstance(n_slices, (int, np.integer)) or n_slices < 1 or
            not np.issubdtype(patch_slice.dtype, np.integer) or
            (patch_slice < 0).any() or (patch_slice >= n_slices).any() or
            not np.isfinite(clean_patch_area).all() or (clean_patch_area < 0).any() or
            (clean_patch_area != np.floor(clean_patch_area)).any()):
        raise ValueError('Valid slice indices and finite nonnegative pixel counts required')
    area = np.bincount(patch_slice[reference], weights=clean_patch_area[reference], minlength=n_slices)
    use_proposal = area == 0
    kept = np.where(use_proposal[patch_slice], proposal, reference)
    return kept, use_proposal

def component_support(inputs):
    """Assign whole-scan regions and their target-free detector support scores."""
    forbidden = ('gt_area', 'target', 'intersection', 'patch_intersection', 'target_mask')
    if any(name in inputs for name in forbidden):
        raise ValueError('Targets must be removed before inference')
    n_patches = len(inputs['patch'])
    if not np.isfinite(inputs['patch']).all() or not np.isfinite(inputs['image']).all():
        raise ValueError('Finite detector probabilities are required')
    if ((inputs['patch'] < 0) | (inputs['patch'] > 1)).any() or ((inputs['image'] < 0) | (inputs['image'] > 1)).any():
        raise ValueError('Detector probabilities must be in [0,1]')
    if (inputs['patch_area'] <= 0).any():
        raise ValueError('Predicted patches must contain foreground pixels')
    component_of_patch = np.full(n_patches, -1, np.int64)
    study_of_component, component_slices, component_patch_ids = [], [], []
    cursor = 0
    for study, rows in enumerate(inputs['studies']):
        if not np.array_equal(inputs['slice_ids'][rows], np.arange(len(rows))):
            raise ValueError('Complete consecutive examination slices are required')
        labels, count = ndimage.label(inputs['occupied'][rows], structure=np.ones((3, 3, 3), bool))
        row_to_z = np.full(len(inputs['area']), -1, int)
        row_to_z[rows] = np.arange(len(rows))
        ids = np.flatnonzero(np.isin(inputs['patch_slice'], rows))
        z = row_to_z[inputs['patch_slice'][ids]]
        y, x = inputs['coords'][ids].T
        assigned = labels[z, y, x]
        if (assigned <= 0).any():
            raise ValueError('Predicted patch is absent from the occupancy grid')
        component_of_patch[ids] = cursor+assigned-1
        for local in range(1, count+1):
            members = ids[assigned == local]
            if not len(members):
                raise ValueError('Occupancy component has no predicted patches')
            component_patch_ids.append(members)
            component_slices.append(np.unique(inputs['patch_slice'][members]))
            study_of_component.append(study)
        cursor += count
    if (component_of_patch < 0).any():
        raise ValueError('Some predicted patches lack examination context')
    if cursor == 0:
        return dict(component_of_patch=component_of_patch, core=np.zeros(0, bool),
            permissive=np.zeros(0, bool), scores={name: np.empty(0) for name in SCORE_NAMES},
            study_of_component=np.zeros(0, int), patch_seed_count=np.zeros(0, int),
            ensemble_seed_count=np.zeros(0, int))
    patch_seed_slices = source_seeds(inputs, 'patch_vote')
    ensemble_seed_slices = source_seeds(inputs, 'ensemble')
    scores = {name: np.empty(cursor, np.float64) for name in SCORE_NAMES}
    core, permissive = np.zeros(cursor, bool), np.zeros(cursor, bool)
    patch_seed_count, ensemble_seed_count = np.zeros(cursor, int), np.zeros(cursor, int)
    for component, members in enumerate(component_patch_ids):
        rows = inputs['patch_slice'][members]
        weights = inputs['patch_area'][members]
        good = inputs['patch'][members] <= .5
        patch_seed = good & patch_seed_slices[rows]
        ensemble_seed = good & ensemble_seed_slices[rows]
        patch_count = len(np.unique(rows[patch_seed]))
        ensemble_count = len(np.unique(rows[ensemble_seed]))
        patch_seed_count[component], ensemble_seed_count[component] = patch_count, ensemble_count
        permissive[component], core[component] = patch_count >= 1, ensemble_count >= 2
        image_success = 1-inputs['image'][rows]
        patch_success = 1-inputs['patch'][members]
        image_mean = float(np.average(image_success, weights=weights))
        patch_mean = float(np.average(patch_success, weights=weights))
        max_image = float(image_success.max())
        unique_rows = component_slices[component]
        scores['image_success_max'][component] = max_image
        scores['image_success_area_mean'][component] = image_mean
        scores['patch_success_area_mean'][component] = patch_mean
        scores['patch_success_max'][component] = float(patch_success.max())
        scores['patch_seed_area_fraction'][component] = float(weights[patch_seed].sum()/weights.sum())
        scores['patch_seed_slice_fraction'][component] = patch_count/len(unique_rows)
        scores['seeded_slice_count'][component] = patch_count
        scores['slice_span'][component] = int(inputs['slice_ids'][unique_rows].max()-inputs['slice_ids'][unique_rows].min()+1)
        scores['joint_geometric_support'][component] = np.sqrt(patch_mean*max_image)
    if np.any(core & ~permissive):
        raise ValueError('Ensemble core must be contained in patch-seeded support')
    return dict(component_of_patch=component_of_patch, core=core, permissive=permissive,
        scores=scores, study_of_component=np.asarray(study_of_component),
        patch_seed_count=patch_seed_count, ensemble_seed_count=ensemble_seed_count)

def source_seeds(inputs, source):
    if source == 'ensemble':
        return slice_accept(inputs, dict(kind='ensemble', cutoff=.8, weighting='uniform',
                                        image_cutoff=.5, concepts=[310, 55]))
    if source == 'patch_vote':
        return slice_accept(inputs, dict(kind='patch_vote', cutoff=.8, weighting='uniform'))
    if source == 'image':
        return slice_accept(inputs, dict(kind='image', image_cutoff=.5))
    raise ValueError('Unknown seed source')

def slice_accept(inputs, rule):
    active = inputs['area'] > 0
    kind = rule['kind']
    if kind == 'baseline':
        return active.copy()
    if kind == 'image':
        return active & (inputs['image'] <= rule['image_cutoff'])
    n = len(active)
    weights = np.ones(len(inputs['patch_area'])) if rule['weighting'] == 'uniform' else inputs['patch_area']
    denominator = np.bincount(inputs['patch_slice'], weights=weights, minlength=n)
    failed = inputs['patch'] > .5
    numerator = np.bincount(inputs['patch_slice'][failed], weights=weights[failed], minlength=n)
    vote = active & (numerator <= rule['cutoff']*denominator)
    if kind == 'patch_vote':
        return vote
    if kind != 'ensemble':
        raise ValueError('Expected a slice rule')
    all_count, last_count = rule['concepts']
    return vote & ((inputs['image'] <= rule['image_cutoff']) |
                   (inputs['active_concepts'] >= all_count) | (inputs['active_l11'] >= last_count))

def geometry_features(inputs, groups):
    """Raw predicted geometry only, including complete-examination neighbors."""
    coords, weights = np.asarray(inputs['coords']), np.asarray(inputs['patch_area'])
    if (not np.isfinite(weights).all() or (weights <= 0).any() or coords.shape != (len(weights), 2)):
        raise ValueError('Finite positive predicted patch areas and aligned coordinates required')
    values = np.zeros((len(groups['sample_slice']), len(GEOMETRY_NAMES)), np.float32)
    for component in np.unique(groups['sample_component']):
        samples = np.flatnonzero(groups['sample_component'] == component)
        members = np.concatenate([groups['members'][sample] for sample in samples])
        area, xy = weights[members], coords[members]
        center = np.average(xy, weights=area, axis=0)
        extent = np.ptp(xy, axis=0)+1
        z = inputs['slice_ids'][groups['sample_slice'][samples]]
        if len(np.unique(z)) != len(z):
            raise ValueError('A component must belong to one examination')
        span = int(np.ptp(z))+1
        local_area = {int(zz): weights[groups['members'][sample]].sum() for zz, sample in zip(z, samples)}
        for zz, sample in zip(z, samples):
            ids = groups['members'][sample]
            a, local = weights[ids], coords[ids]
            size, midpoint = np.ptp(local, axis=0)+1, np.average(local, weights=a, axis=0)
            values[sample] = (np.log1p(a.sum()), np.log1p(len(ids)), *list(midpoint/63),
                *list(size/64), len(ids)/np.prod(size), a.sum()/(256*len(ids)),
                np.log1p(area.sum()), np.log1p(len(members)), np.log1p(len(samples)),
                np.log1p(span), len(samples)/span, *list(extent/64), a.sum()/area.sum(),
                len(ids)/len(members), (zz-z.min())/max(span-1, 1),
                np.linalg.norm(midpoint-center)/64,
                local_area.get(int(zz)-1, 0)/max(a.sum(), local_area.get(int(zz)-1, 0), 1),
                local_area.get(int(zz)+1, 0)/max(a.sum(), local_area.get(int(zz)+1, 0), 1))
    if not np.isfinite(values).all():
        raise ValueError('Invalid region geometry')
    return values

def partition(inputs, component_of_patch):
    """One sample per component/slice; no learned decisions or labels enter."""
    if any(key in inputs for key in ('gt_area', 'target', 'target_mask', 'patch_intersection')):
        raise ValueError('Region inputs must be target-free')
    row, component = np.asarray(inputs['patch_slice']), np.asarray(component_of_patch)
    n = len(inputs['area'])
    if (row.ndim != 1 or row.shape != component.shape or not len(row) or
            not np.issubdtype(component.dtype, np.integer) or (component < 0).any() or
            not np.issubdtype(row.dtype, np.integer) or (row < 0).any() or (row >= n).any()):
        raise ValueError('Aligned nonempty valid component and slice indices required')
    pairs, inverse = np.unique(np.column_stack((component, row)), axis=0, return_inverse=True)
    return dict(sample_component=pairs[:, 0], sample_slice=pairs[:, 1], patch_sample=inverse,
        members=[np.flatnonzero(inverse == sample) for sample in range(len(pairs))])

def calibrated_keep(inputs, point, topology=None):
    support = seed_support(inputs, point, topology)
    accepted = support['core'] | (support['permissive'] &
                (support['success'] >= point['region_success_cutoff']))
    return accepted[support['component_of_patch']]

def seed_support(inputs, point, topology=None):
    """Compute distinct-slice seed support without labels or concept refitting."""
    if any(key in inputs for key in ('gt_area', 'target', 'target_mask',
                                    'intersection', 'patch_intersection')):
        raise ValueError('Targets must be removed before inference')
    patch, image = np.asarray(inputs['patch']), np.asarray(inputs['image'])
    if not all(np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all() for a in (patch, image)):
        raise ValueError('Finite probabilities in [0,1] are required')
    patch_cut, image_cut, region_cut = [point[key] for key in SPEC_KEYS[2:]]
    if not (np.isfinite([patch_cut, image_cut, region_cut]).all() and
            0 <= patch_cut <= 1 and 0 <= image_cut <= 1 and 0 <= region_cut <= 1.01):
        raise ValueError('Invalid probability cutoffs')
    if point['concept_count_policy'] not in ('enabled', 'disabled'):
        raise ValueError('Unknown concept-count policy')
    topology = component_support(inputs) if topology is None else topology
    component = topology['component_of_patch']
    if component.shape != patch.shape or (component < 0).any():
        raise ValueError('Invalid component membership')
    n = len(inputs['area'])
    count = int(component.max())+1 if len(component) else 0
    rows = inputs['patch_slice']
    total = np.bincount(rows, minlength=n)
    failed = np.bincount(rows, weights=patch > patch_cut, minlength=n)
    voted = (inputs['area'] > 0) & (total > 0) & (failed <= .8*total)
    verified = image <= image_cut
    if point['concept_count_policy'] == 'enabled':
        verified |= (inputs['active_concepts'] >= 310) | (inputs['active_l11'] >= 55)
    ensemble = voted & verified
    good = patch <= patch_cut
    def distinct_slices(accepted):
        pairs = np.unique(component[accepted]*n+rows[accepted])
        return np.bincount(pairs//n, minlength=count)
    patch_count = distinct_slices(good & voted[rows])
    ensemble_count = distinct_slices(good & ensemble[rows])
    success = np.zeros(count, float)
    np.maximum.at(success, component, 1-patch)
    core, permissive = ensemble_count >= 2, patch_count >= 1
    if np.any(core & ~permissive):
        raise ValueError('Core must be a subset of permissive support')
    return dict(component_of_patch=component, core=core, permissive=permissive,
        patch_seed_count=patch_count, ensemble_seed_count=ensemble_count,
        success=success, patch_vote=voted, ensemble=ensemble)
