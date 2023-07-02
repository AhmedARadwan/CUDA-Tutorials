#include <ros/ros.h>
#include <autoware_msgs/DetectedObjectArray.h>
#include <chrono>
#include <thrust/copy.h>

struct Transformation {
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;
    double droll = 0.0;
    double dpitch = 0.0;
    double dyaw = 0.0;
};

struct Quaternion {
    double w;
    double x;
    double y;
    double z;
};

// CUDA transformation function
__global__ void cudaTransformObject(autoware_msgs::DetectedObject* cuda_object_data, Transformation m_transformation, Quaternion dquaternion, int object_size) {
    int index = threadIdx.x;

    if (index < object_size) {
        autoware_msgs::DetectedObject* object = &cuda_object_data[index];

        Quaternion quaternion;
        quaternion.x = object->pose.orientation.x;
        quaternion.y = object->pose.orientation.y;
        quaternion.z = object->pose.orientation.z;
        quaternion.w = object->pose.orientation.w;

        // Calculate new quaternion orientation
        Quaternion T_quaternion;

        T_quaternion.w = dquaternion.w * quaternion.w - dquaternion.x * quaternion.x - dquaternion.y * quaternion.y - dquaternion.z * quaternion.z;
        T_quaternion.x = dquaternion.w * quaternion.x + dquaternion.x * quaternion.w + dquaternion.y * quaternion.z - dquaternion.z * quaternion.y;
        T_quaternion.y = dquaternion.w * quaternion.y - dquaternion.x * quaternion.z + dquaternion.y * quaternion.w + dquaternion.z * quaternion.x;
        T_quaternion.z = dquaternion.w * quaternion.z + dquaternion.x * quaternion.y - dquaternion.y * quaternion.x + dquaternion.z * quaternion.w;

        // Update transformed position and orientation (quaternion)
        object->pose.position.x = object->pose.position.x + m_transformation.dx;
        object->pose.position.y = object->pose.position.y + m_transformation.dy;
        object->pose.position.z = object->pose.position.z + m_transformation.dz;
        object->pose.orientation.x = T_quaternion.x;
        object->pose.orientation.y = T_quaternion.y;
        object->pose.orientation.z = T_quaternion.z;
        object->pose.orientation.w = T_quaternion.w;
    }
}

class ObjectTransformer {
public:
    ObjectTransformer() {
        // Initialize CUDA device
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        ROS_INFO("Using CUDA device: %s", deviceProp.name);

        // Initialize ROS node handle
        nh_ = ros::NodeHandle("~");

        // Subscribe to the input detected object array topic
        object_array_sub_ = nh_.subscribe("/tracking/vehicles", 10, &ObjectTransformer::objectArrayCallback, this);

        // Advertise the output transformed object array topic
        transformed_array_pub_ = nh_.advertise<autoware_msgs::DetectedObjectArray>("/tracking/vehicles/transformed", 10);

        m_transformation.dx = 45.0;
        m_transformation.dy = -12.1;
        m_transformation.dz = -0.3;
        m_transformation.droll = 0.07;
        m_transformation.dpitch = -0.01;
        m_transformation.dyaw = 3.075;

        // Calculate quaternion components
        const double cy = cos(m_transformation.dyaw * 0.5);
        const double sy = sin(m_transformation.dyaw * 0.5);
        const double cp = cos(m_transformation.dpitch * 0.5);
        const double sp = sin(m_transformation.dpitch * 0.5);
        const double cr = cos(m_transformation.droll * 0.5);
        const double sr = sin(m_transformation.droll * 0.5);

        dquaternion.w = cy * cp * cr + sy * sp * sr;
        dquaternion.x = cy * cp * sr - sy * sp * cr;
        dquaternion.y = sy * cp * sr + cy * sp * cr;
        dquaternion.z = sy * cp * cr - cy * sp * sr;

        transformed_objects_.resize(0);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber object_array_sub_;
    ros::Publisher transformed_array_pub_;

    Transformation m_transformation;
    Quaternion dquaternion;

    std::vector<autoware_msgs::DetectedObject> transformed_objects_;

    void objectArrayCallback(const autoware_msgs::DetectedObjectArray::ConstPtr& msg) {
        auto startTime = std::chrono::high_resolution_clock::now();

        const std::vector<autoware_msgs::DetectedObject>& objects = msg->objects;
        int object_size = objects.size();

        // Allocate device memory for the object data
        autoware_msgs::DetectedObject* cuda_object_data;
        cudaMalloc(&cuda_object_data, object_size * sizeof(autoware_msgs::DetectedObject));

        // Copy the object data to the device memory
        cudaMemcpy(cuda_object_data, objects.data(), object_size * sizeof(autoware_msgs::DetectedObject), cudaMemcpyHostToDevice);

        // Launch the CUDA transformation kernel
        int num_threads = object_size;
        cudaTransformObject<<<1, num_threads>>>(cuda_object_data, m_transformation, dquaternion, object_size);

        // Copy the transformed object data back to the host memory
        autoware_msgs::DetectedObject* transformed_objects;
        cudaMallocHost(&transformed_objects, object_size * sizeof(autoware_msgs::DetectedObject));

        cudaMemcpy(transformed_objects, cuda_object_data, object_size * sizeof(autoware_msgs::DetectedObject), cudaMemcpyDeviceToHost);

        auto endTime = std::chrono::high_resolution_clock::now();
        double transformation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1000000.0;
        std::cout << "transformation time: " << transformation_time << "\n";

        // Create and publish the transformed object array message
        autoware_msgs::DetectedObjectArray transformed_msg;
        transformed_msg.header = msg->header;
        thrust::copy(transformed_objects, transformed_objects + object_size, std::back_inserter(transformed_msg.objects));
        transformed_array_pub_.publish(transformed_msg);

        // Free device memory
        cudaFree(cuda_object_data);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "object_transformer_node");

    ObjectTransformer transformer;

    ros::spin();

    return 0;
}

