import rclpy
import threading
from rclpy.node import Node
import math

from px4_msgs.msg import VehicleOdometry, TimesyncStatus, SensorCombined, VehicleAttitude, VehicleCommand
from geometry_msgs.msg import PoseStamped

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np


class Imu_odo(Node):

    def __init__(self):
        super().__init__('imu_odometry')

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,depth=10)

        # Publishers
        self.create_subscription(SensorCombined,"/fmu/out/sensor_combined",self.pose_estimation,qos_profile)
        self.create_subscription(TimesyncStatus,"/fmu/out/timesync_status",self.time_update,qos_profile)
        #self.create_subscription(VehicleAttitude,"/fmu/out/vehicle_attitude",self.quat_update,qos_profile)
        self.create_subscription(VehicleOdometry,"/fmu/in/vehicle_mocap_odometry",self.mocap_update,qos_profile)
        self.odometry_pub = self.create_publisher(VehicleOdometry,"/fmu/in/vehicle_visual_odometry",10)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand,'/fmu/in/vehicle_command',qos_profile)

        print('subscriber line done')

        # Timer at 50Hz
        self.timer = self.create_timer(0.02, self.timer_callback)

        self.orientation =np.array([1.0,0.0,0.0,0.0])

        # curr state initialization
        self.curr_pose =np.array([0.0,0.0,0.0])
        self.old_pose = np.array([0.0,0.0,0.0])
        self.curr_velocity =np.array([0.0,0.0,0.0])
        self.old_velocity = np.array([0.0,0.0,0.0])
        self.curr_accl =np.array([0.0,0.0,0.0])

        self.last_timestamp=0
        self.timestamp=0
        self.bias = np.array([0.0,0.0,0.0])
        self.accel_bias = np.array([0.0,0.0,0.0])
        self.gyro_rad_bias = np.array([0.0,0.0,0.0])
        self.bias_gyro = np.array([0.0,0.0,0.0])
        
        self.covar_matrix = np.eye(9)*0.01
        #self.covar_matrix[9:12,9:12] = np.eye(3)*5.0
        #self.covar_matrix[12:15,12:15] = np.eye(3)*0.5

        
        # process noise
        self.process_noise = np.eye(9)
        self.process_noise[0:3,0:3] = np.eye(3)*1e-8 # position
        self.process_noise[3:6,3:6] = np.eye(3)*1e-4 # velocity
        self.process_noise[6:9,6:9] = np.eye(3)*1e-8 # orientation
        #self.process_noise[9:12,9:12] = np.eye(3)*1e-7 # acceleration bias
        #self.process_noise[12:15,12:15] = np.eye(3)*1e-8 # gyro bias

        #mocap pose
        self.mocap_pose = np.array([0.0,0.0,0.0])
        self.mocap_velocity = np.array([0.0,0.0,0.0])
        self.old_mocapPose = np.array([0.0,0.0,0.0])

        self.velo_counter = 0
        self.counter = 0.0
        self.comput_bias=True
        self.initialization = True
        self.initialization_counter = 0
        self.mocap_last_timestamp = 0

    def time_update(self,msg):
        self.timestamp=msg.timestamp

    def quat_update(self, msg):
        self.orientation = msg.q

    def mocap_update(self, msg):

        if self.mocap_last_timestamp==0:
            self.mocap_last_timestamp=msg.timestamp
            dt = 0
            self.mocap_initial_time=self.mocap_last_timestamp

        else:
            dt = (msg.timestamp-self.mocap_last_timestamp)*1e-6
            self.mocap_last_timestamp=msg.timestamp

        
        self.mocap_pose[0] = msg.position[0]
        self.mocap_pose[1] = msg.position[1]
        self.mocap_pose[2] = msg.position[2]

        self.orientation[0]=msg.q[0]
        self.orientation[1]=msg.q[1]
        self.orientation[2]=msg.q[2]
        self.orientation[3]=msg.q[3]
        
        if dt>0:
            self.mocap_velocity = (self.mocap_pose-self.old_mocapPose)/dt
        else:
            pass

        self.old_mocapPose = self.mocap_pose.copy()


    def pose_estimation(self, msg):
        #print('subscriber call back triggered')
        # sensor data

        self.gyro_rad = msg.gyro_rad
        gyro_dt = msg.gyro_integral_dt*1e-6

        acceleration = msg.accelerometer_m_s2
        accl_dt = msg.accelerometer_integral_dt*1e-6

        if self.last_timestamp==0:
            self.last_timestamp=msg.timestamp
            dt = 0
            self.initial_time=self.last_timestamp

        else:
            dt = (msg.timestamp-self.last_timestamp)*1e-6
            self.last_timestamp=msg.timestamp

        #print(f'gyro readings before bias removal are {self.gyro_rad} and gyro bias are {self.gyro_rad_bias}')                
        #acceleration= acceleration-self.accel_bias
        self.gyro_rad = self.gyro_rad - self.gyro_rad_bias
        #print(f'gyro readings after bias removal are {self.gyro_rad} and gyro bias are {self.gyro_rad_bias}') 

        gyro_norm = np.linalg.norm(self.gyro_rad)

        rot_mat = self.quat_2_rot_mat(self.orientation)
        R = rot_mat.T
        global_accl = R@acceleration
        gravity = np.array([0.0,0.0,-9.81])
        self.curr_accl = global_accl-gravity
        self.curr_accl = self.curr_accl - self.accel_bias


        if (msg.timestamp-self.initial_time)*1e-6<=5.0:
            self.bias = self.bias + self.curr_accl
            self.bias_gyro = self.bias_gyro + self.gyro_rad
            self.counter = self.counter+1.0
        else:
            if self.comput_bias:
                self.accel_bias = self.bias/self.counter
                self.gyro_rad_bias = self.bias_gyro/self.counter
                self.comput_bias=False
            else:                
                #self.curr_accl= self.curr_accl-self.accel_bias
                #pass

                drone_angle = self.gyro_rad*dt
                

                gyro_norm = np.linalg.norm(self.gyro_rad)
                self.curr_velocity = self.curr_velocity + self.curr_accl*dt
                self.curr_pose =self.curr_pose+self.curr_velocity*dt + 0.5*self.curr_accl*dt*dt

                print(f'position before update is {self.curr_pose}')

                self.pose_prediction =self.curr_pose +self.curr_velocity*dt
                self.velo_prediction = self.curr_velocity+self.curr_accl*dt
                self.gyro_prediction = drone_angle +self.gyro_rad*dt

                measurement_vector = np.array([[self.curr_pose[0]],
                                            [self.curr_pose[1]],
                                            [self.curr_pose[2]],
                                            [self.curr_velocity[0]],
                                            [self.curr_velocity[1]],
                                            [self.curr_velocity[2]],
                                            [drone_angle[0]],
                                            [drone_angle[1]],
                                            [drone_angle[2]],
                                            ])

                # prediction_vector = np.array([[self.pose_prediction[0]],
                #                               [self.pose_prediction[1]],
                #                               [self.pose_prediction[2]],
                #                               [self.velo_prediction[0]],
                #                               [self.velo_prediction[1]],
                #                               [self.velo_prediction[2]],
                #                               [self.gyro_prediction[0]],
                #                               [self.gyro_prediction[1]],
                #                               [self.gyro_prediction[2]],
                #                               ])

                # state_vector = np.array([[self.curr_pose[0]],
                #                          [self.curr_pose[1]],
                #                          [self.curr_pose[2]],
                #                          [self.curr_velocity[0]],
                #                          [self.curr_velocity[1]],
                #                          [self.curr_velocity[2]]])
                #                         #  [drone_angle[0]],
                #                         #  [drone_angle[1]],
                #                         #  [drone_angle[2]]])


                state_vector = np.array([[self.curr_pose[0]],
                                         [self.curr_pose[1]],
                                         [self.curr_pose[2]],
                                         [self.curr_velocity[0]],
                                         [self.curr_velocity[1]],
                                         [self.curr_velocity[2]],
                                         [self.accel_bias[0]],
                                         [self.accel_bias[1]],
                                         [self.accel_bias[2]]])




                # prediction_vector = np.array([[self.pose_prediction[0]],
                #                             [self.pose_prediction[1]],
                #                             [self.pose_prediction[2]]])

                measurement_vector = np.array([[self.mocap_pose[0]],
                                            [self.mocap_pose[1]],
                                            [self.mocap_pose[2]],
                                            [self.mocap_velocity[0]],
                                            [self.mocap_velocity[1]],
                                            [self.mocap_velocity[2]]
                                            ])
                
                state_transition_mat =np.eye(9)

                #position dependency on velocity
                state_transition_mat[0:3,3:6] = np.eye(3)*dt
                state_transition_mat[0:3,6:9] = -0.5*R@np.eye(3)*dt*dt
                state_transition_mat[3:6,6:9] = -np.eye(3)*dt

                #state_transition_mat[0:3,6:9] = -0.5*R@self.skew(self.curr_accl)*dt*dt
                # state_transition_mat[0:3,9:12] = -0.5*R*dt*dt
                #state_transition_mat[3:6,6:9] = -R@self.skew(self.curr_accl)*dt
                # state_transition_mat[3:6,9:12]= -R*dt
                # state_transition_mat[6:9,12:15]= -np.eye(3)*dt
                # state_transition_mat[6:9,6:9] = np.eye(3) - self.skew(self.gyro_rad)*dt


                # update covariance - sigmat = Gt*sigmat*Gt' + Rt
                self.covar_matrix = state_transition_mat@self.covar_matrix@state_transition_mat.T + self.process_noise*dt

                #H 3x9
                measurement_matrix = np.zeros((6,9))
                measurement_matrix[0:3,0:3]=np.eye(3)
                measurement_matrix[3:6,3:6]=np.eye(3)

                #9x1
                #error = prediction_vector - measurement_matrix@measurement_vector
                error =  measurement_vector- measurement_matrix@state_vector

                
                measurement_noise = np.zeros((6,6))
                measurement_noise[0:3,0:3] = np.eye(3)*1e-3
                measurement_noise[3:6,3:6] = np.eye(3)*1e-5

                #error covar 3x3 
                error_covar = measurement_matrix @ self.covar_matrix @ measurement_matrix.T + measurement_noise

                #Kalman gain 9x3
                K = self.covar_matrix @measurement_matrix.T @ np.linalg.inv(error_covar)
                #print(f'acceleration is{self.curr_accl}')
                #print(f'measurement vector {measurement_vector}')
                #print(f'prediction vector is {prediction_vector}')
                
                #print(f'Eror covar matrix is {error}')
                #print(f'kalman gain is {K}')
                #print(f'Kalman gain multiply by error is {K@error}')

                state_vector = state_vector + K @ error
                #state = K @ error
                #dx = state.T

                #print(f'position before update is {self.curr_pose}')
                # print(f'state error is {dx}')
                # print(f'position state error is {dx[0:3]}')


                # #print(f'position before update is {self.curr_pose}')
                # print(f'state error is {dx.shape}')
                # print(f'position state error is {dx[0:3].shape}')



                I =np.eye(9)
                self.covar_matrix = (I -K @ measurement_matrix ) @ self.covar_matrix
                
                self.curr_pose = state_vector[0:3,0]#np.array([measurement_vector[0,0],measurement_vector[1,0],measurement_vector[2,0]])
                self.curr_velocity = state_vector[3:6,0]#np.array([measurement_vector[3,0],measurement_vector[4,0],measurement_vector[5,0]])
                print(f'position after update is {self.curr_pose}')
                self.accel_bias = state_vector[6:9,0]


                updated_acceleration= (self.curr_velocity-self.old_velocity)/dt
                #self.accel_bias = updated_acceleration-self.curr_accl
                #self.accel_bias = self.curr_accl-updated_acceleration

                #self.old_velocity = self.curr_velocity.copy()
        

    def timer_callback(self):
        pose = VehicleOdometry()
        pose.timestamp = self.timestamp
        pose.quality = 100
        pose.pose_frame = VehicleOdometry.POSE_FRAME_FRD
        pose.velocity_frame = VehicleOdometry.VELOCITY_FRAME_FRD

        # if np.allclose(self.old_mocapPose,self.mocap_pose):
        #     print(f"mocap pose is {self.mocap_pose}")
        # else:
        #     # self.curr_pose[0]=self.mocap_pose[0]
        #     # self.curr_pose[1]=self.mocap_pose[1]
        #     # self.curr_pose[2]=self.mocap_pose[2]
        #     print(f'old mocap pose is {self.old_mocapPose}')
        #     # self.old_mocapPose = self.mocap_pose
        #     print(f'current mocap pose is {self.mocap_pose}')
            


        pose.position = [self.curr_pose[0],self.curr_pose[1],self.curr_pose[2]]

        pose.q[0] = self.orientation[0]
        pose.q[1] = self.orientation[1]
        pose.q[2] = self.orientation[2]
        pose.q[3] = self.orientation[3]

        pose.velocity[0] = self.curr_velocity[0]
        pose.velocity[1] = self.curr_velocity[1]
        pose.velocity[2] = self.curr_velocity[2]

        print(f'current position in publisher is {pose.position}')


        self.odometry_pub.publish(pose)

    def quat_mult(self,q1,q2):
        qw1,qx1,qy1,qz1 = q1
        qw2,qx2,qy2,qz2 = q2
        
        qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2
        qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
        qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
        qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2

        q = np.array([qw,qx,qy,qz])

        return q
    
    def quat_2_rot_mat(self,q):
        w,x,y,z= q
        Rot_mat = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z),2*(x*z+w*y)],
                           [2*(x*y+w*z),1-2*(x*x+z*z), 2*(y*z-w*x)],
                           [2*(x*z-w*y), 2*(y*z+w*x),1-2*(x*x+y*y)]])
        

        return Rot_mat
    
    def skew(self,v):
        return np.array([[0.0,-v[2],v[1]],
                         [v[2],0.0,-v[0]],
                         [-v[1],v[0],0.0]])


    def publish_vehicle_command(
            self,
            command,
            param1=0.0,
            param2=0.0
    
        ):
    
            msg = VehicleCommand()
    
            msg.timestamp = self.timestamp
    
            msg.param1 = param1
            msg.param2 = param2
    
            msg.command = command
    
            msg.target_system = 1
            msg.target_component = 1
    
            msg.source_system = 1
            msg.source_component = 1
    
            msg.from_external = True
    
            self.vehicle_command_pub.publish(msg)


def main():

    rclpy.init()

    node = Imu_odo()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
