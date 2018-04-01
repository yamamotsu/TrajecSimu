#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:29:08 2018

@author: shugo
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

#TODO: Make PostProcess & ResultPlot class

"""
# class for post-processing
"""
    
class PostProcess_dist():
    
    # ------------------------------
    # method for setup landing distribution coordinate
    # ------------------------------
    def set_coordinate_izu(self):
        # !!!! hardcoding for 2018 izu ura-sabaku
        # Set limit range in maps (Defined by North latitude and East longitude)
        point_rail = np.array([34.735972, 139.420944])
        point_switch = np.array([34.735390, 139.421377])
        point_tent = np.array([34.731230, 139.423150])
        point_range = np.array([[34.735715,	139.420922],
                                [34.731750,	139.421719],
                                [34.733287,	139.424590],
                                [34.736955,	139.426038],
                                [34.738908,	139.423597],
                                [34.740638,	139.420681],
                                [34.741672,	139.417387],
                                [34.735715,	139.420922],
                                ])
    
        # Define convert value from lat/long[deg] to meter[m]
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0
    
        # Convert from absolute coordinate to relative coordinate
        point_rail = point_rail - origin
        point_switch = point_switch - origin
        point_tent = point_tent - origin
        point_range = point_range - origin
    
        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2)
        self.xy_tent = np.zeros(2)
        self.xy_range = np.zeros([point_range[:,0].size, 2])
    
        self.xy_switch[1] = lat2met * point_switch[0]
        self.xy_switch[0] = lon2met * point_switch[1]
        self.xy_tent[1] = lat2met * point_tent[0]
        self.xy_tent[0] = lon2met * point_tent[1]
        self.xy_range[:,1] = lat2met * point_range[:,0]
        self.xy_range[:,0] = lon2met * point_range[:,1]
    
        # Set magnetic declination
        mag_dec = -7.53   # [deg] @ Izu
        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])
    
        # Rotate by magnetic declination angle
        self.xy_switch = mat_rot @ self.xy_switch
        self.xy_tent = mat_rot @ self.xy_tent
    
        for i in range(self.xy_range[:,0].size):
            self.xy_range[i,:] = mat_rot @ self.xy_range[i,:]
    
        return None
    
    
    
    def judge_in_range(self, drop_point):
    
        dir_pat = drop_point.size[1]
        vel_pat = drop_point.size[2]
    
        cross_num = np.zeros([1, dir_pat, vel_pat])
    
        judge_result = np.mod(cross_num, 2)
    
        return judge_result
    
    
    # ------------------------------
    # method for plot map and landing points
    # ------------------------------
    def plot_map(self):
    
        # Set limit range in maps
        lim_radius = 50.0   # define circle limit area
        self.set_coordinate_izu()
    
        """ # for tamura version
        # Set map image
        img_map = Image.open("./map/Izu_map_mag.png")
        img_list = np.asarray(img_map)
        img_height = img_map.size[0]
        img_width = img_map.size[1]
        img_origin = np.array([722, 749])    # TODO : compute by lat/long of launcher point
    
        #pixel2meter = (139.431463 - 139.41283)/1800.0 * lon2met
        pixel2meter = 0.946981208125
        """
        
        # for extended version
        # Set map image
        img_map = Image.open("./map/map_extended.png")
        img_list = np.asarray(img_map)
        img_height = img_map.size[1]
        img_width = img_map.size[0]
        img_origin = np.array([231, 284])    # TODO : compute by lat/long of launcher point
        
        pixel2meter = 7.796379
        # """
    
        # Define image range 
        img_left =   -1.0 * img_origin[0] * pixel2meter
        img_right = (img_width - img_origin[0]) * pixel2meter
        img_top = img_origin[1] * pixel2meter
        img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter
    
        plt.figure(figsize=(12,10))
        plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))
    
        # plot setting
        ax = plt.axes()
        color_line = '#ffff33'    # Yellow
        color_circle = 'r'    # Red
    
        # Set circle object
        cir_rail = patches.Circle(xy=self.xy_rail, radius=lim_radius, ec=color_circle, fill=False)
        cir_switch = patches.Circle(xy=self.xy_switch, radius=lim_radius, ec=color_circle, fill=False)
        cir_tent = patches.Circle(xy=self.xy_tent, radius=lim_radius, ec=color_circle, fill=False)
        ax.add_patch(cir_rail)
        ax.add_patch(cir_switch)
        ax.add_patch(cir_tent)
    
        # Write landing permission range
        plt.plot(self.xy_rail[0], self.xy_rail[1], '.', color=color_circle)
        plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
        plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
        plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)
    
        ax.set_aspect('equal')
    
    
    def plot_sct(self, drop_point, wind_speed_array, launcher_elev_angle, fall_type):
        # -------------------
        # plot landing distribution
        # hardcoded for izu ura-sabaku
        # 
        # INPUT: 
        #        drop_point: (n_speed * n_angle * 2(xy) ndarry): landing point coordinate
        #        wind_speed_array: array of wind speeds 
        #        lancher_elev_angle: elevation angle [deg]
        #        fall_type = 'Parachute' or 'Ballistic'
        # 
        # -------------------
    
        # plot map 
        self.plot_map()
    
        # file existence check
        #file_exist = os.path.exists("./output")
    
        #if file_exist == False:
        #    os.mkdir("./output")
    
        title_name = fall_type + ", Launcher elev. " + str(int(launcher_elev_angle)) + " deg"
    
    
        imax = len(wind_speed_array)
        for i in range(imax):
    
            # cmap = plt.get_cmap("winter")
    
            labelname = str(wind_speed_array[i]) + " m/s"
            plt.plot(drop_point[i,:,0],drop_point[i,:,1], label = labelname, linewidth=2, color=cm.Oranges(i/imax))
            
            
        # output_name = "output/Figure_elev_" + str(int(rail_elev)) + ".png"
        output_name = 'Figure_' + fall_type + '_elev' + str(int(launcher_elev_angle)) + 'deg.eps'
    
        plt.title(title_name)
        plt.legend()
        plt.savefig(output_name, bbox_inches='tight')
        plt.show()
        
        
"""
# class for auto-judge
"""
class JudgeInside():
    
    def __init__(self, xy_range, xy_center=None, lim_radius=50.0):
        # INPUT:
        #   xy_range = (n*2) ndarray: configure range by n surrounding points
        #   xy_center = (m*2) ndarray: configure m circle center coordinates
        #   lim_radius = limit radius from the specified points above
        
        
        # print("Judge inside : ON")
        
        # setup!
        # Check range area is close or not
        if np.allclose(xy_range[0,:], xy_range[-1,:]):
            #print("")
            #print("Range is close.")
            #print("")

            self.xy_range = xy_range

        else:
            print("")
            print("Range area is not close.")
            print("Connect first point and last point automatically.")
            print("")

            point_first = xy_range[0,:]
            self.xy_range = np.vstack((xy_range, point_first))

        self.xy_center = xy_center
        self.lim_radius = lim_radius


    def judge_inside(self, check_point):

        # Check limit circle area is defined
        if self.xy_center is None:
            circle_flag = False
        else:
            circle_flag = True            


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
        
        # print('judge!,',  judge_result, check_point)

        return judge_result    


if __name__ == '__main__':
    tmp = PostProcess_dist()
    tmp.set_coordinate_izu()
    tmp.plot_map()

    # drop_point_test = 

# END IF 


    