#include <ros/ros.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <chrono>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2/LinearMath/Quaternion.h>


class ObjectTransformer
{
    public:
        ObjectTransformer()
        {
            // Initialize ROS node handle
            nh_ = ros::NodeHandle("~");

            // Subscribe to the input detected object array topic
            object_array_sub_ = nh_.subscribe("/tracking/vehicles", 10, &ObjectTransformer::objectArrayCallback, this);

            // Advertise the output transformed object array topic
            transformed_array_pub_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/tracking/vehicles/transformed", 10);

            // adding objects 2 to output
            transform.header.stamp = ros::Time::now();
            transform.header.frame_id = "local";
            transform.child_frame_id = "local2";
            transform.transform.translation.x = 45.0;
            transform.transform.translation.y = -12.1;
            transform.transform.translation.z = -0.3;
            tf2::Quaternion q;
            q.setRPY(0.07, -0.01, 3.075);

            q=q.normalize();
            transform.transform.rotation.x = q[0];
            transform.transform.rotation.y = q[1];
            transform.transform.rotation.z = q[2];
            transform.transform.rotation.w = q[3];

            
            
            
        }

    private:
        ros::NodeHandle nh_;
        ros::Subscriber object_array_sub_;
        ros::Publisher transformed_array_pub_;
        geometry_msgs::TransformStamped transform;


        void objectArrayCallback(const autoware_msgs::DetectedObjectArray::ConstPtr& msg)
        {
            
            autoware_msgs::DetectedObjectArray tmp_objects = *msg;
            auto startTime = std::chrono::high_resolution_clock::now();
            autoware_msgs::DetectedObjectArray out_objects;

            // transform objects back to original frame
            geometry_msgs::Pose tmp_pose;
            for (auto& object : tmp_objects.objects){
                tf2::doTransform(object.pose, tmp_pose, transform);
                object.pose = tmp_pose;
                out_objects.objects.push_back(object);

            }
            auto endTime = std::chrono::high_resolution_clock::now();
            double transformation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count()/1000000.0;
            std::cout << "transformation time: " << transformation_time << "\n";

            out_objects.header.stamp = msg->header.stamp;
            out_objects.header.frame_id = "combined";
            transformed_array_pub_.publish(out_objects);
        }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_transformer_node");

    ObjectTransformer transformer;

    ros::spin();

    return 0;
}
