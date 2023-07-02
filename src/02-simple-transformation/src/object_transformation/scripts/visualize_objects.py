#!/usr/bin/python
import rospy
from autoware_msgs.msg import DetectedObjectArray
from visualization_msgs.msg import MarkerArray, Marker
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import numpy as np

classes = {
    "car"                   : 0,
    "truck"                 : 1,
    "construction_vehicle"  : 2,
    "bus"                   : 3,
    "trailer"               : 4,
    "barrier"               : 5,
    "motorcycle"            : 6,
    "bicycle"               : 7,
    "pedestrian"            : 8,
    "traffic_cone"          : 9
}
target_classes = {
    "car"                   : 0,
    "truck"                 : 1,
    "construction_vehicle"  : 2,
    "bus"                   : 3,

}

def callback(data):
    print("callback!")
    marker_array = MarkerArray()
    bbox_msg = BoundingBoxArray()
    bbox_msg.header = data.header
    for i, bounding_box in enumerate(data.objects):
        if bounding_box.label in target_classes:
            marker = Marker()
            marker.header = data.header
            marker.ns = "labels"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose = bounding_box.pose
            marker.pose.position.z += 5
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.text = bounding_box.label + "\nID: " + str(bounding_box.id)
                        
            marker_array.markers.append(marker)

            tmp_box = BoundingBox()
            tmp_box.header = data.header
            tmp_box.pose = bounding_box.pose
            tmp_box.pose.position.z = -2.5
            tmp_box.dimensions = bounding_box.dimensions
            tmp_box.value = bounding_box.score
            bbox_msg.boxes.append(tmp_box)
    pub.publish(marker_array)
    pub_bboxes.publish(bbox_msg)



rospy.init_node('objects_visualizer_publisher')
sub = rospy.Subscriber("/input", DetectedObjectArray, callback)
pub = rospy.Publisher("/debug/marker", MarkerArray, queue_size=10)
pub_bboxes = rospy.Publisher("/debug/boxes", BoundingBoxArray, queue_size=10)
rospy.spin()
