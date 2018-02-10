import os
import math
import numpy as np
import numpy.linalg as nplin


class JudgeInside():
    def __init__(self):
        print("Judge inside : ON")


    def set_limit_area(self, xy_range):

        # Check range area is close or not
        if np.allclose(xy_range[0,:], xy_range[-1,:]):
            print("")
            print("Range is close.")
            print("")

            self.xy_range = xy_range

        else:
            print("")
            print("Range area is not close.")
            print("Connect first point and last point automatically.")
            print("")

            point_first = xy_range[0,:]
            self.xy_range = np.vstack((xy_range, point_first))


    def set_limit_circle(self, xy_center, lim_radius = 50.0):

        self.xy_center = xy_center
        self.lim_radius = lim_radius


    def judge_inside(self, check_point):

        # Check limit circle area is defined
        try:
            self.xy_center
            circle_flag = True

        except AttributeError:
            circle_flag = False


        # Initialize count of line cross number
        cross_num = 0

        # Count number of range area
        point_num = self.xy_range.shape[0]

        # Judge inside or outside by cross number
        for point in range(point_num - 1):

            point_ymin = np.min(self.xy_range[point:point+2, 1])
            point_ymax = np.max(self.xy_range[point:point+2, 1])

            if check_point[1] == self.xy_range[point, 1]:

                if check_point[0] < self.xy_range[point, 0]:
                    cross_num += 1

                else:
                    pass

            elif point_ymin < check_point[1] < point_ymax:

                dx = self.xy_range[point+1, 0] - self.xy_range[point, 0]
                dy = self.xy_range[point+1, 1] - self.xy_range[point, 1]

                if dx == 0.0:
                    # Line is parallel to y-axis
                    judge_flag = self.xy_range[point, 1] - check_point[1]

                elif dy == 0.0:
                    # Line is parallel to x-axis
                    judge_flag = -1.0

                else:
                    # y = ax + b (a:slope,  b:y=intercept)
                    slope = dy / dx
                    y_intercept = self.xy_range[point, 1] - slope * self.xy_range[point, 0] 

                    # left:y,  right:ax+b
                    left_eq = check_point[1]
                    right_eq = slope * check_point[0] + y_intercept

                    judge_flag = slope * (left_eq - right_eq)


                if judge_flag > 0.0:
                    # point places left side of line 
                    cross_num += 1.0

                elif judge_flag < 0.0:
                    # point places right side of line
                    pass

            else:
                pass

        # odd number : inside,  even number : outside
        judge_result = np.mod(cross_num, 2)
        
        # check judge circle mode
        if circle_flag == True:

            center_num = self.xy_center.shape[0]

            for center in range(center_num):
                # Compute distance between drop_point and center of limit circle
                length_point = np.sqrt((check_point[0] - self.xy_center[center, 0])**2 + \
                                       (check_point[1] - self.xy_center[center, 1])**2)

                # Judge in limit circle or not
                if length_point <= self.lim_radius:
                    judge_result = np.bool(False)

                else:
                    pass

        else:
            pass

        # Convert from float to bool (True:inside,  False:outside)
        judge_result = np.bool(judge_result)

        return judge_result
