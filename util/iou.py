import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely import geometry


def calc_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    iou = 0.0
    if box1[2] <= box1[0] or box1[3] <= box1[1]:
        return iou
    if box2[2] <= box2[0] or box2[3] <= box2[1]:
        return iou      
    if box1[2] <= box2[0] or box1[0] >= box2[2]:
        return iou
    if box1[3] <= box2[1] or box1[1] >= box2[3]:
        return iou

    xl_min = min(box1[0], box2[0])
    xl_max = max(box1[0], box2[0])
    xr_min = min(box1[2], box2[2])
    xr_max = max(box1[2], box2[2])

    yl_min = min(box1[1], box2[1])
    yl_max = max(box1[1], box2[1])
    yr_min = min(box1[3], box2[3])
    yr_max = max(box1[3], box2[3])

    inter = float(xr_min-xl_max)*float(yr_min-yl_max)
    union = float(xr_max-xl_min)*float(yr_max-yl_min)

    iou = float(inter) / float(union)
    if iou < 0:
        iou = 0.0
    return iou


def calc_iou_multiple(boxes1, boxes2):
    # TODO: DOC
    poly1 = poly_union(boxes1)
    poly2 = poly_union(boxes2)

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    iou = float(inter) / float(union)
    iou = max(iou, 0.0) # underflow?

    return iou


def poly_union(boxes):
    # TODO: DOC
    res_poly = geometry.Polygon()
    for box in boxes:
        rec = geometry.box(box[0], box[1], box[2], box[3])
        res_poly = res_poly.union(rec)
    return res_poly


def exact_group_union(boxes):
    """
    Args:
        boxes: list of [xmin, ymin, xmax, ymax]

        p1 *-----
           |     |
           |_____* p2
    Returns: list of non-overlapping groups where each group is a list of boxes, within each group, boxes have a non-zero overlap with each other
        
    """

    n = len(boxes)

    # Construct adjacency matrix based on overlap
    overlaps = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if calc_iou(boxes[i], boxes[j]) > 0.0:
                overlaps[i,j] = 1

    # Find disjoint sets
    graph = csr_matrix(overlaps)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    groups = []
    for i in range(n_components):
        group = [boxes[j] for j in range(n) if labels[j]==i]
        groups.append(group)

    return groups


def rec_convex_hull_union(boxes):
    # TODO: DOC
    x = np.array(boxes)
    mins = x[:,:2].min(axis=0)
    maxs = x[:,2:4].max(axis=0)
    union = np.concatenate((mins, maxs), axis=0)
    return list(union)

