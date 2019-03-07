#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

from scipy import spatial
import numpy as np

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.base_waypoints = None
        self.base_waypoints_2d = None
        self.waypoints_kdtree = None
        self.pose = None
        self.stopline_wd_idx = -1

        # TODO: Add other member variables you need below

        # rospy.spin()
        self.loop()
    
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # nearest_waypoint_idx = self.get_nearest_waypoint_idx()
                self.publish_waypoints()
            rate.sleep()

    def pose_cb(self, msg):
        # TODO: Implement
        
        self.pose = msg
        
    def get_nearest_waypoint_idx(self):

        cur_x = self.pose.pose.position.x
        cur_y = self.pose.pose.position.y

        nearest_waypoint_idx = self.waypoints_kdtree.query([cur_x, cur_y], 1)[1]

        nearst_coord = self.base_waypoints_2d[nearest_waypoint_idx]
        prev_coord = self.base_waypoints_2d[nearest_waypoint_idx-1]

        nearest_vect = np.array(nearst_coord)
        prev_vect = np.array(prev_coord)
        cur_vect = np.array([cur_x, cur_y])

        val = np.dot(nearest_vect-prev_vect, cur_vect-nearest_vect)

        if val > 0:
            nearest_waypoint_idx = (nearest_waypoint_idx + 1) % len(self.base_waypoints_2d)

        return nearest_waypoint_idx

    def waypoints_cb(self, waypoints):
        # TODO: Implement

        self.base_waypoints = waypoints
        if not self.base_waypoints_2d:
            self.base_waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_kdtree = spatial.KDTree(self.base_waypoints_2d)
        
    def publish_waypoints(self):
        # lane = Lane()
        # lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[nearest_waypoint_idx:nearest_waypoint_idx+LOOKAHEAD_WPS]
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)
        
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wd_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass
    
    def generate_lane(self):
        lane = Lane()

        nearest_idx = self.get_nearest_waypoint_idx()
        farthest_idx = nearest_idx + LOOKAHEAD_WPS

        base_waypoints = self.base_waypoints.waypoints[nearest_idx:farthest_idx]

        if self.stopline_wd_idx == -1 or (self.stopline_wd_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, nearest_idx)
        
        return lane
    
    def decelerate_waypoints(self, waypoints, nearest_idx):
        temp = []

        for i, wp in enumerate(waypoints):
            p = Waypoint()

            p.pose = wp.pose

            stop_idx = max(self.stopline_wd_idx - nearest_idx - 2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)

            if vel < 1:
                vel = 0
            
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        
        return temp



    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
