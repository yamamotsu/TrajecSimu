#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:29:08 2018

@author: shugo
"""

# import os
import subprocess
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import sympy.geometry as sg

#TODO: Make PostProcess & ResultPlot class

"""
# class for post-processing
"""


class PostProcess_dist():

    def __init__(self, loc):
        # get launch location: 'izu' or 'noshiro_sea'
        self.launch_location = loc

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
        # END FOR

        """
        # -------------------------------
        # hardcoding: actual landing point of Felix-yayaHeavy on 3/23
        point_land = np.array([ 34.73534332, 139.4215288] )
        # switch 34.735390, 139.421377
        # rail   34.735972, 139.420944
        point_land -= origin
        self.xy_land = np.zeros(2)
        self.xy_land[1] = lat2met * point_land[0]
        self.xy_land[0] = lon2met * point_land[1]
        self.xy_land = mat_rot @ self.xy_land
        print('actual landing point xy:', self.xy_land)
        # -------------------------------
        """

        return None




    def set_coordinate_noshiro(self):
        # !!!! hardcoding for 2018 noshiro umi_uchi
        # Set limit range in maps (Defined by North latitude and East longitude)

        # -----------------------------------
        #  Define permitted range here
        # -----------------------------------

        #used as "outside_centers" & "outside_radius", meaning NOT drop inside the circle
        point_rail = np.array([40.242865, 140.010450])
        point_switch = np.array([40.240382, 140.009295])
        point_tent = np.array([40.2414098, 140.0100285])
        self.lim_radius = 50.0

        #used as "inside_center" & "inside_radius", meaning MUST drop inside the circle
        self.hachiya_radius = 1500.0 # [m]
        point_center = np.array([40.245567,	139.993297]) #center of circle

        #used as two points of "over_line", meaning MUST drop over the line
        point_point = np.array([[40.23665, 140.00579],
                                [40.25126, 140.00929],
                                ])

        # Set magnetic declination
        mag_dec = -8.9  # [deg] @ noshiro
        
        #to add if necessary
        #used as "under_line", meaning MUST drop under the line
        # point_point2 = np.array([[40.23665, 140.00579],
        #                        [40.27126, 140.00929],
        #                       ])

        # -------- End definition --------

        # Define convert value from lat/long[deg] to meter[m]
        origin = point_rail
        earth_radius = 6378150.0    # [km]
        lat2met = 2 * math.pi * earth_radius / 360.0
        lon2met = 2 * math.pi * earth_radius * np.cos(np.deg2rad(origin[0])) / 360.0

        # Convert from absolute coordinate to relative coordinate
        point_rail = point_rail - origin
        point_switch = point_switch - origin
        point_tent = point_tent - origin
        point_point = point_point - origin
        point_center = point_center - origin

        # Convert from lat/long to meter (ENU coordinate)
        self.xy_rail = np.zeros(2)
        self.xy_switch = np.zeros(2)
        self.xy_tent = np.zeros(2)
        self.xy_point = np.zeros([point_point[:,0].size, 2])
        self.xy_center = np.zeros(2)

        self.xy_switch[1] = lat2met * point_switch[0]
        self.xy_switch[0] = lon2met * point_switch[1]
        self.xy_tent[1] = lat2met * point_tent[0]
        self.xy_tent[0] = lon2met * point_tent[1]
        self.xy_point[:,1] = lat2met * point_point[:,0] #y of all
        self.xy_point[:,0] = lon2met * point_point[:,1] #x of all
        self.xy_center[1] = lat2met * point_center[0]
        self.xy_center[0] = lon2met * point_center[1]
        self.xy_rail[1] = lat2met * point_rail[0]
        self.xy_rail[0] = lon2met * point_rail[1]
        
        mag_dec = np.deg2rad(mag_dec)
        mat_rot = np.array([[np.cos(mag_dec), -1 * np.sin(mag_dec)],
                            [np.sin(mag_dec), np.cos(mag_dec)]])

        # Rotate by magnetic declination angle
        self.xy_switch = mat_rot @ self.xy_switch
        self.xy_tent = mat_rot @ self.xy_tent
        self.xy_center = mat_rot @ self.xy_center

        for i in range(self.xy_point[:,0].size):
            self.xy_point[i,:] = mat_rot @ self.xy_point[i,:]

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

        if self.launch_location == 'izu':
            # Set limit range in maps
            lim_radius = 50.0   # define circle limit area
            self.set_coordinate_izu()

            # for tamura version
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


            fig = plt.figure(figsize=(12,10))

            # plot setting
            ax = fig.add_subplot(111)
            color_line = '#ffff33'    # Yellow
            color_circle = 'r'    # Red

            # Set circle object

            cir_rail = patches.Circle(xy=self.xy_rail, radius=lim_radius, ec=color_circle, fill=False)
            cir_switch = patches.Circle(xy=self.xy_switch, radius=lim_radius, ec=color_circle, fill=False)
            cir_tent = patches.Circle(xy=self.xy_tent, radius=lim_radius, ec=color_circle, fill=False)
            ax.add_patch(cir_rail)
            ax.add_patch(cir_switch)
            ax.add_patch(cir_tent)

            # plot map
            plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

            # Write landing permission range
            plt.plot(self.xy_rail[0], self.xy_rail[1], 'r.', color=color_circle, markersize = 12)
            plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
            plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
            plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)

            """
            # plot landing point for 2018/3/23
            plt.plot(self.xy_land[0], self.xy_land[1], 'r*', markersize = 12, label='actual langing point')
            """

        #for NOSHIRO SEA!!
        elif self.launch_location == 'noshiro_sea':
              # Set limit range in maps
              self.set_coordinate_noshiro()

              # Set map image
              img_map = Image.open("./map/noshiro_new_rotate.png")
              img_list = np.asarray(img_map)
              img_height = img_map.size[1]
              # print(img_map.size)
              img_width = img_map.size[0]
              img_origin = np.array([894, 647])    # TODO : compute by lat/long of launcher point

              #pixel2meter
              pixel2meter = 8.96708

              # Define image range
              img_left =   -1.0 * img_origin[0] * pixel2meter
              img_right = (img_width - img_origin[0]) * pixel2meter
              img_top = img_origin[1] * pixel2meter
              img_bottom = -1.0 * (img_height - img_origin[1]) * pixel2meter

              #calculate intersections of "inside_circle" and "over_line"
              center1 = sg. Point(self.xy_center[0],self.xy_center[1])
              radius1 = self.hachiya_radius
              circle1 = sg.Circle(center1,radius1)
              line = sg.Line(sg.Point(self.xy_point[0,0],self.xy_point[0,1]), sg.Point(self.xy_point[1,0],self.xy_point[1,1]))
              result1 = sg.intersection(circle1, line)
              intersection1_1 = np.array([float(result1[0].x), float(result1[0].y)])
              intersection1_2 = np.array([float(result1[1].x), float(result1[1].y)])

              #caluculate equation of hachiya_line(="over_line")
              self.a = (self.xy_point[1,1]-self.xy_point[0,1])/(self.xy_point[1,0]-self.xy_point[0,0])
              self.b = (self.xy_point[0,1]*self.xy_point[1,0]-self.xy_point[1,1]*self.xy_point[0,0])/(self.xy_point[1,0]-self.xy_point[0,0])
              self.x = np.arange(intersection1_1[0],intersection1_2[0],1)
              self.y = self.a*self.x + self.b
              self.hachiya_line = np.array([self.a, self.b])

              # plot setting
              plt.figure(figsize=(10,10))
              ax = plt.axes()
              color_line = '#ffff33'    # Yellow
              color_circle = 'r'    # Red

              # Set circle object
              cir_rail = patches.Circle(xy=self.xy_rail, radius=self.lim_radius, ec=color_line, fill=False)
              #cir_switch = patches.Circle(xy=self.xy_switch, radius=self.lim_radius, ec=color_circle, fill=False)
              #cir_tent = patches.Circle(xy=self.xy_tent, radius=self.lim_radius, ec=color_circle, fill=False)
              cir_center = patches.Circle(xy=self.xy_center, radius=self.hachiya_radius, ec=color_circle, fill=False)

              ax.add_patch(cir_rail)
              #ax.add_patch(cir_switch)
              #ax.add_patch(cir_tent)
              ax.add_patch(cir_center)
              
              # plot map
              plt.imshow(img_list, extent=(img_left, img_right, img_bottom, img_top))

              # Write landing permission range
              plt.plot(self.x, self.y,"r")
              plt.plot(self.xy_rail[0], self.xy_rail[1], '.', color=color_circle)
              #plt.plot(self.xy_switch[0], self.xy_switch[1], '.', color=color_circle)
              #plt.plot(self.xy_tent[0], self.xy_tent[1], '.', color=color_circle)
              #plt.plot(self.xy_range[:,0], self.xy_range[:,1], '--', color=color_line)
              plt.plot(self.xy_center[0], self.xy_center[1], '.', color=color_circle)

        else:
              print('Error!! Available location is: izu or noshiro_sea' )

        # ax.set_aspect('equal')

        return None

    def plot_sct(self, drop_point, wind_speed_array, launcher_elev_angle, fall_type):
        # -------------------
        # plot landing distribution
        # hardcoded for noshiro
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
        output_name = 'results/Figure_' + fall_type + '_elev' + str(int(launcher_elev_angle)) + 'deg.eps'

        plt.title(title_name)
        plt.legend()
        plt.savefig(output_name, bbox_inches='tight')
        plt.show()


"""
# class for auto-judge
"""
class JudgeInside():

    def __init__(self, input_dict):
        # INPUT:
        #   "input_dict" is dictionary type variable

        # print("Judge inside : ON")

        # setup!
        # Check range area is close or not
        """
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
        """

        self.xy_range = input_dict["range"]
        self.outside_centers = input_dict["outside_centers"]
        self.outside_radius = input_dict["outside_radius"]
        self.over_line = input_dict["over_line"]
        self.under_line = input_dict["under_line"]
        self.inside_center = input_dict["inside_center"]
        self.inside_radius = input_dict["inside_radius"]


    def judge_inside(self, check_point):
        
        # initialize bool for result
        judge_result = True
        
        # check inside circles-------------------------------------
        if self.inside_center is None:
            circle_flag1 = False
        else:
            circle_flag1 = True

        # judge inside the circle
        if circle_flag1 == True:

            center_num = self.inside_center.shape[0]

            for center in range(center_num):
                # Compute distance between drop_point and center of limit circle
                #length_point = np.sqrt((check_point[0] - self.xy_center[center, 0])**2 + \
                #                       (check_point[1] - self.xy_center[center, 1])**2)
                length_point = np.linalg.norm(check_point-self.inside_center)

                # Judge in limit circle or not
                if length_point > self.inside_radius:
                    judge_result = np.bool(False)

                #else:
                #    judge_result = np.bool(True)

        #-------------------------------------------------------------


        # check ourside the circle-----------------------------------
        if self.outside_centers is None:
            circle_flag2 = False
        else:
            circle_flag2 = True

        # Judge outside cirle
        if circle_flag2 == True:

            center_num = self.outside_centers.shape[0]

            for center in range(center_num):
                # Compute distance between drop_point and center of limit circle
                length_point = np.sqrt((check_point[0] - self.outside_centers[center, 0])**2 + \
                                       (check_point[1] - self.outside_centers[center, 1])**2)

                # Judge in limit circle or not
                if length_point <= self.outside_radius:
                    judge_result = np.bool(False)

        #----------------------------------------------------------


        #check under the line--------------------------------------
        if self.under_line is None:
            line_flag1 = False
        else:
            line_flag1 = True

       # Judge under the line
        if line_flag1 == True:

           if check_point[1] > self.under_line[0]*check_point[0]+self.under_line[1]:
            judge_result = np.bool(False)

           #else:
           # judge_result = np.bool(True)

        #----------------------------------------------------------


        #check over the line--------------------------------------
        if self.over_line is None:
            line_flag2 = False
        else:
            line_flag2 = True

       # Judge under the line
        if line_flag2 == True:

           if check_point[1] < self.over_line[0]*check_point[0]+self.over_line[1]:
               judge_result = np.bool(False)

        #-------------------------------------------------------------


        #check inside the range--------------------------------------
        # Initialize count of line cross number
        if self.xy_range is None:
            range_flag = False

        else:
            range_flag = True

       # judge inside the circle
        if range_flag == True:

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

                    #elif judge_flag < 0.0:
                        # point places right side of line
                    #    pass

                    # odd number : inside,  even number : outside
                    if np.mod(cross_num, 2) == 0:
                        # outside of the range. Nogo
                        judge_result = False
                        
                    
                    



        # Convert from float to bool (True:inside,  False:outside)
        # judge_result = np.bool(judge_result)

        # print('judge!,',  judge_result, check_point)

        return judge_result



if __name__ == '__main__':
    tmp = PostProcess_dist('noshiro_sea')
    tmp.set_coordinate_noshiro()
    tmp.plot_map()
                    




         #drop_point_test =
#END IF
