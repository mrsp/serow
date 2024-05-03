#!/usr/bin/env python

import rospy
from geometry_msgs.msg import WrenchStamped

def foot_force_callback_left(msg):
    # Modify the frame_id of the message for left foot
    msg.header.frame_id = "leftFoot"
    # Publish the modified message on the new topic for left foot
    pub_left.publish(msg)

def foot_force_callback_right(msg):
    # Modify the frame_id of the message for right foot
    msg.header.frame_id = "rightFoot"
    # Publish the modified message on the new topic for right foot
    pub_right.publish(msg)

def main():
    rospy.init_node('foot_force_frame_id_changer')

    # Subscribe to the original topics for left and right foot
    rospy.Subscriber('/ihmc_ros/valkyrie/output/foot_force_sensor/left', WrenchStamped, foot_force_callback_left)
    rospy.Subscriber('/ihmc_ros/valkyrie/output/foot_force_sensor/right', WrenchStamped, foot_force_callback_right)

    # Publish the modified messages on the new topics for left and right foot
    global pub_left, pub_right
    pub_left = rospy.Publisher('/ihmc_ros/valkyrie/output/foot_force/left', WrenchStamped, queue_size=10)
    pub_right = rospy.Publisher('/ihmc_ros/valkyrie/output/foot_force/right', WrenchStamped, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    main()
