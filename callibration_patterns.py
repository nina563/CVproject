# square_size in mm
# dimension is in number of squares
# size in mm - full size of the pattern
# x_blank and y_blank in mm  - white spaces around the checkerboard
# u_pair - which lines should be used to detect vanishing point that lies on the right side
# v_pair - which lines should be used to detect vanishing point that lies on the left side
checkered_board = {"dimension": (8, 5), "square_size": 125, "size":(1189, 841), "x_blank": 31, "y_blank": 44}
cam1 = {"start_pair":("D", "C"), "u_pair":(2,3), "v_pair":(0,1)}
cam2 ={"start_pair":("A", "D"), "u_pair":(0,1), "v_pair":(2,3)}
cam3 = {"start_pair":("A", "D"),"u_pair":(0,1), "v_pair":(2,3)}
cam4 ={"start_pair":("B", "A"), "u_pair":(2,3), "v_pair":(0,1)}
cam5 = {"start_pair":("C", "B"), "u_pair":(0,1), "v_pair":(2,3)}
cam6 ={"start_pair":("D", "C"), "u_pair":(2,3), "v_pair":(0,1)}