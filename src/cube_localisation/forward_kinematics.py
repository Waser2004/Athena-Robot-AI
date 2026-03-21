import numpy as np
from math import radians, sin, cos, degrees


# Forward Kinematics after the Denavit-Hartenberg method
class DHForwardKinematics(object):
    def __init__(self, theta, alpha, radius, distance):
        """
        This class is used to perform forward kinematics on a 6DOF Robot arm
        """
        assert len(list(theta)) == len(list(alpha)) == len(list(radius)) == len(
            list(distance)), "theta, alpha, radius, distance must have same amount of arguments"

        # joint angles
        self.joints = list(np.zeros(len(list(theta))))

        # parameters for Denavit-Hartenberg transformation matrix
        self.theta = list(theta)
        self.alpha = list(alpha)
        self.radius = list(radius)
        self.distance = list(distance)

        # list of Denavit-Hartenberg transformation matrices
        self.matrices = []
        for i, th in enumerate(list(theta)):
            self.matrices.append(self.dh_matrix(th, list(alpha)[i], list(radius)[i], list(distance)[i]))

        # results and update state
        self.results = []
        for _ in range(len(self.theta)):
            self.results.append(self.dh_matrix(0, 0, 0, 0))
        self.state = "updated"

    # set the rotations of each joint
    def set_joint_angles(self, *args: float):
        """
        This function is used to set the joint angles of the Robot

        Args:
            args: joint angles
        """
        assert len(args) == len(self.joints), "there must be same amount arguments as joints"

        # assign new joint angles
        for i, joint in enumerate(args):
            self.joints[i] = radians(joint)

        # update state
        self.state = "updated"

    # return rotation and transformation information for required joint to the user
    def get_joint_rot_trans(self, joint_index: int) -> ((float, float, float), (float, float, float)):
        """
        This function calculates the position/rotation of the joint based on the previously given other joint angles

        Args:
            joint_index: the index of the joint from which we want the location/rotation

        Return:
            list representing the position of the joint
            list representing the rotation of the joint
        """
        assert joint_index < len(self.joints), f"index is not in range of {len(self.joints)}"
        
        # update matrices if necessary
        if len(self.results) != len(self.joints) or self.state == "updated":
            self.calculate_forward_kinematics()
        
        # extract rotations and transformations from the Denavit-Hartenberg matrices
        rotation = self.get_rot_from_matrix(self.results[joint_index])
        transformation = self.results[joint_index][0, 3], self.results[joint_index][1, 3], self.results[joint_index][2, 3]
        return rotation, transformation

    # return rotation matrix for required joint to the user
    def get_joint_rotation_matrix(self, joint_index: int):
        """
        This function calculates the rotation of the joint based on the previously given other joint angles

        Args:
            joint_index: the index of the joint from which we want the location/rotation

        Return:
            list representing the rotation matrix of the joint
        """
        assert joint_index < len(self.joints), f"index is not in range of {len(self.joints)}"

        # update matrices if necessary
        if len(self.results) != len(self.joints) or self.state == "updated":
            self.calculate_forward_kinematics()

        return self.results[joint_index][:3, :3]

    def get_joint_translation_vector(self, joint_index: int):
        """
        This function calculates the translation vector of the joint based on the previously given other joint angles

        Args:
            joint_index: the index of the joint from which we want the location/rotation

        Return:
            list representing the translation vector of the joint
        """
        assert joint_index < len(self.joints), f"index is not in range of {len(self.joints)}"

        # update matrices if necessary
        if len(self.results) != len(self.joints) or self.state == "updated":
            self.calculate_forward_kinematics()

        return self.results[joint_index][:3, 3]

    # return full transform matrix for required joint to the user
    def get_joint_transform(self, joint_index: int):
        """
        Return the full 4x4 transform of the joint in base coordinates.
        """
        assert joint_index < len(self.joints), f"index is not in range of {len(self.joints)}"

        # update matrices if necessary
        if len(self.results) != len(self.joints) or self.state == "updated":
            self.calculate_forward_kinematics()

        # Ensure a plain ndarray (not numpy.matrix)
        return np.asarray(self.results[joint_index], dtype=float)
    
    # calculate the result matrices for the Denavit-Hartenberg forward kinematics
    def calculate_forward_kinematics(self):
        """
        This function performs the forward kinematics
        """
        # update 
        if self.state == "updated":
            self.matrices.clear()
            for i, th in enumerate(self.theta):
                self.matrices.append(self.dh_matrix(th+self.joints[i], self.alpha[i], self.radius[i], self.distance[i]))
            self.state = "not updated"

        # multiply matrices
        for i, matrix in enumerate(self.matrices):
            if i == 0:
                self.results[0] = matrix
            if i > 0:
                self.results[i] = self.results[i-1] @ matrix

    # definition method of Denavit-Hartenberg transformation matrix
    @staticmethod
    def dh_matrix(o, ar, r, d):
        """
        This function defines the dh matrix
        """
        return np.matrix([[cos(o), -sin(o) * cos(ar), sin(o) * sin(ar), r * cos(o)],
                          [sin(o), cos(o) * cos(ar), -cos(o) * sin(ar), r * sin(o)],
                          [0, sin(ar), cos(ar), d],
                          [0, 0, 0, 1]])

    # extracts rotations from matrix
    @staticmethod
    def get_rot_from_matrix(r):
        """
        This Functino converts a rotation matrix into x, y, z rotation values
        """
        sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(r[2, 1], r[2, 2])
            y = np.arctan2(-r[2, 0], sy)
            z = np.arctan2(r[1, 0], r[0, 0])
        else:
            x = np.arctan2(-r[1, 2], r[1, 1])
            y = np.arctan2(-r[2, 0], sy)
            z = 0

        return degrees(x), degrees(y), degrees(z)


class RobotFKModel(DHForwardKinematics):
    def __init__(self):
        # Forward Kinematics setup
        pri_arm_origin = [[0], [0], [163.2]]

        pri_arm_length = [[0], [0], [236.5]]
        sec_arm_length = [[0], [97.5], [236.5]]
        ter_arm_length = [[0], [255], [0]]

        theta = [
            0, 
            radians(90), 
            radians(90),
            0, 
            radians(-90), 
            0
        ]
        alpha = [
            radians(90),
            0,
            radians(-90),
            radians(90),
            radians(-90),
            0
        ]
        radius = [
            0, 
            pri_arm_length[2][0], 
            sec_arm_length[1][0], 
            0, 
            0, 
            0
        ]
        distance = [
            pri_arm_origin[2][0],
            0,
            0,
            sec_arm_length[2][0],
            0,
            ter_arm_length[1][0]
        ]

        super().__init__(theta, alpha, radius, distance)
    