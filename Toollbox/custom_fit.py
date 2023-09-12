import math

def point_coord_fit( pop:list ) -> float:
    
    data_points =   [[0,0],[77, 68], [12, 75], [32, 17], [51, 64],[20, 19], [72, 87], [80, 37], [35, 82], [2, 15],[18, 90], [33, 50], [85, 52], [97, 27], [37, 67],[20, 82], [49, 0], [62, 14], [7, 60], [100, 100]]
    finnal = []
    # Iterate through each pair of points in the list
    for data in pop:
        data = [0] + list(data) + [19]
        total_length = 0.0
        for i in range(1, len(data)):
            
            x1, y1 = data_points[data[i - 1]]
            x2, y2 = data_points[data[i]]
            
            # Calculate the length of the line segment between (x1, y1) and (x2, y2)
            segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Add the segment length to the total length
            total_length += segment_length
    
        finnal.append(total_length)
    min_val = min(finnal)
    return _scale_values(finnal), min_val

def _scale_values(original_list):
    min_v = min(original_list)
    max_v = max(original_list)
    
    if min_v == max_v:
        max_v += 10
    
    scaled_list = [ (( x  - min_v)/(max_v - min_v )) * (0.0 - 1.0) + 1.0  for x in original_list ]
    
    
    return scaled_list
