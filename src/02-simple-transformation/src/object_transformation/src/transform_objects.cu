#include <ros/ros.h>
#include <autoware_msgs/DetectedObjectArray.h>


// CUDA transformation function
__global__ void cudaTransformObject(autoware_msgs::DetectedObject* cuda_object_data, const float* transformation_matrix, int object_size)
{
    int index = threadIdx.x;
    // printf("DEBUG: index: %d, object_size: %d\n", index, object_size);

    if (index < object_size)
    {
        autoware_msgs::DetectedObject* object = &cuda_object_data[index];

        // Extract position and orientation (quaternion)
        float x = object->pose.position.x;
        float y = object->pose.position.y;
        float z = object->pose.position.z;
        float qx = object->pose.orientation.x;
        float qy = object->pose.orientation.y;
        float qz = object->pose.orientation.z;
        float qw = object->pose.orientation.w;

        // Apply transformation to position
        float transformed_x = transformation_matrix[0] * x + transformation_matrix[1] * y + transformation_matrix[2] * z + transformation_matrix[3];
        float transformed_y = transformation_matrix[4] * x + transformation_matrix[5] * y + transformation_matrix[6] * z + transformation_matrix[7];
        float transformed_z = transformation_matrix[8] * x + transformation_matrix[9] * y + transformation_matrix[10] * z + transformation_matrix[11];

        // Apply transformation to quaternion orientation
        float transformed_qx = transformation_matrix[0] * qx + transformation_matrix[1] * qy + transformation_matrix[2] * qz + transformation_matrix[3];
        float transformed_qy = transformation_matrix[4] * qx + transformation_matrix[5] * qy + transformation_matrix[6] * qz + transformation_matrix[7];
        float transformed_qz = transformation_matrix[8] * qx + transformation_matrix[9] * qy + transformation_matrix[10] * qz + transformation_matrix[11];
        float transformed_qw = transformation_matrix[12] * qx + transformation_matrix[13] * qy + transformation_matrix[14] * qz + transformation_matrix[15] * qw;

        // Update transformed position and orientation (quaternion)
        object->pose.position.x = transformed_x;
        object->pose.position.y = transformed_y;
        object->pose.position.z = transformed_z;
        object->pose.orientation.x = transformed_qx;
        object->pose.orientation.y = transformed_qy;
        object->pose.orientation.z = transformed_qz;
        object->pose.orientation.w = transformed_qw;
    }
}

class ObjectTransformer
{
public:
    ObjectTransformer()
    {
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
        
        // Allocate host memory for the transformation matrix
        cudaMallocHost(&host_transformation_matrix, 16 * sizeof(float));

        // Modify host_transformation_matrix with your desired transformation values
        // Translation values (x, y, z)
        host_transformation_matrix[0] = 20.0;
        host_transformation_matrix[1] = 0.0;
        host_transformation_matrix[2] = 0.0;

        // Rotation values (roll, pitch, yaw)
        float roll = 0.0;  // Convert roll to radians
        float pitch = 0.0;  // Convert pitch to radians
        float yaw = 1.57;  // Convert yaw to radians

        float cos_roll = cos(roll);
        float sin_roll = sin(roll);
        float cos_pitch = cos(pitch);
        float sin_pitch = sin(pitch);
        float cos_yaw = cos(yaw);
        float sin_yaw = sin(yaw);

        // Rotation matrix
        host_transformation_matrix[3] = cos_yaw * cos_pitch;
        host_transformation_matrix[4] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll;
        host_transformation_matrix[5] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll;

        host_transformation_matrix[6] = sin_yaw * cos_pitch;
        host_transformation_matrix[7] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll;
        host_transformation_matrix[8] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll;

        host_transformation_matrix[9] = -sin_pitch;
        host_transformation_matrix[10] = cos_pitch * sin_roll;
        host_transformation_matrix[11] = cos_pitch * cos_roll;

        host_transformation_matrix[12] = 0.0;
        host_transformation_matrix[13] = 0.0;
        host_transformation_matrix[14] = 0.0;
        host_transformation_matrix[15] = 1.0;
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber object_array_sub_;
    ros::Publisher transformed_array_pub_;
    float* host_transformation_matrix;


    void objectArrayCallback(const autoware_msgs::DetectedObjectArray::ConstPtr& msg)
    {
        // Extract the detected objects from the received message
        const std::vector<autoware_msgs::DetectedObject>& objects = msg->objects;
        int object_size = objects.size();

        // Allocate device memory for the object data
        autoware_msgs::DetectedObject* cuda_object_data;
        cudaMalloc(&cuda_object_data, object_size * sizeof(autoware_msgs::DetectedObject));

        // Copy the object data to the device memory
        cudaMemcpy(cuda_object_data, objects.data(), object_size * sizeof(autoware_msgs::DetectedObject), cudaMemcpyHostToDevice);

        // Launch the CUDA transformation kernel
        int num_threads = object_size;
        cudaTransformObject<<<1, num_threads>>>(cuda_object_data, host_transformation_matrix, object_size);
        cudaDeviceSynchronize();

        // Copy the transformed object data back to the host memory
        autoware_msgs::DetectedObject* transformed_objects;
        cudaMallocHost(&transformed_objects, object_size * sizeof(autoware_msgs::DetectedObject));

        cudaMemcpy(transformed_objects, cuda_object_data, object_size * sizeof(autoware_msgs::DetectedObject), cudaMemcpyDeviceToHost);

        // Create and publish the transformed object array message
        autoware_msgs::DetectedObjectArray transformed_msg;
        transformed_msg.header = msg->header;
        for (int i=0 ; i < object_size ; i++){
            transformed_msg.objects.push_back(transformed_objects[i]);
        }
        transformed_array_pub_.publish(transformed_msg);

        // Free device memory
        cudaFree(cuda_object_data);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "object_transformer_node");

    ObjectTransformer transformer;

    ros::spin();

    return 0;
}
