import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from tf_transformations import euler_from_quaternion

import math
import numpy as np


def sgn(num):
    if num >= 0:
        return 1
    else:
        return -1


def pt_to_pt_distance(pt1, pt2):
    return np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)


class PurePursuit(Node):
    def __init__(self):
        super().__init__("pure_pursuit_controller")
        self.odom_subscriber = self.create_subscription(
            Odometry, "odom", self.odom_callback, 10
        )
        self.path_subscriber = self.create_subscription(
            Path, "plan", self.path_callback, 10
        )
        self.cmd_vel_publisher = self.create_publisher(Twist, "ro_vel", 10)
        self.timer = self.create_timer(50 / 1000, self.timer_callback)

        self.odom = Odometry()
        self.path = Path()

        self.last_found_index = 0

        self.path = Path()

    def odom_callback(self, msg):
        self.odom = msg

    def path_callback(self, msg):
        self.path = msg
        self.last_found_index = 0

    def point_from_path(self, idx):
        return [
            self.path.poses[idx].pose.position.x,
            self.path.poses[idx].pose.position.y,
        ]

    def timer_callback(self):

        tolerance = 0.1
        lookAheadDis = 0.2
        linearVel = 2

        # extract currentX and currentY
        currentX = self.odom.pose.pose.position.x
        currentY = self.odom.pose.pose.position.y

        if (
            not self.path.poses
            or pt_to_pt_distance([currentX, currentY], self.point_from_path(-1))
            < tolerance
        ):
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd_vel)
            print("no path or goal reached: stopping")
            return

        # extract orientation
        _, _, currentHeading = euler_from_quaternion(
            [
                self.odom.pose.pose.orientation.x,
                self.odom.pose.pose.orientation.y,
                self.odom.pose.pose.orientation.z,
                self.odom.pose.pose.orientation.w,
            ]
        )
        currentHeading *= 180 / np.pi

        # use for loop to search intersections
        intersectFound = False
        startingIndex = self.last_found_index

        # init to current loc for no reason
        goalPt = [currentX, currentY]

        for i in range(startingIndex, len(self.path.poses) - 1):

            # beginning of line-circle intersection code
            x1 = self.point_from_path(i)[0] - currentX
            y1 = self.point_from_path(i)[1] - currentY
            x2 = self.point_from_path(i + 1)[0] - currentX
            y2 = self.point_from_path(i + 1)[1] - currentY
            dx = x2 - x1
            dy = y2 - y1
            dr = math.sqrt(dx**2 + dy**2)
            D = x1 * y2 - x2 * y1
            discriminant = (lookAheadDis**2) * (dr**2) - D**2

            print(
                "checking between points (p1, p2, D): ",
                [x1, y1],
                [x2, y2],
                discriminant,
            )

            if discriminant >= 0:
                print("D>0: intersects")
                sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
                sol_y1 = (-D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
                sol_y2 = (-D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

                sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
                sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
                # end of line-circle intersection code

                minX = min(self.point_from_path(i)[0], self.point_from_path(i + 1)[0])
                minY = min(self.point_from_path(i)[1], self.point_from_path(i + 1)[1])
                maxX = max(self.point_from_path(i)[0], self.point_from_path(i + 1)[0])
                maxY = max(self.point_from_path(i)[1], self.point_from_path(i + 1)[1])

                # if one or both of the solutions are in range
                if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or (
                    (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                ):
                    print("at least one in range")

                    foundIntersection = True

                    # if both solutions are in range, check which one is better
                    if (
                        (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)
                    ) and (
                        (minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)
                    ):
                        print("checking better soln")
                        # make the decision by compare the distance between the intersections and the next point in path
                        if pt_to_pt_distance(
                            sol_pt1, self.point_from_path(i + 1)
                        ) < pt_to_pt_distance(sol_pt2, self.point_from_path(i + 1)):
                            goalPt = sol_pt1
                        else:
                            goalPt = sol_pt2

                    # if not both solutions are in range, take the one that's in range
                    else:
                        print("only one in range")
                        # if solution pt1 is in range, set that as goal point
                        if (minX <= sol_pt1[0] <= maxX) and (
                            minY <= sol_pt1[1] <= maxY
                        ):
                            print("soln 1 in range")
                            goalPt = sol_pt1
                        else:
                            print("soln 2 in range")
                            goalPt = sol_pt2

                    # only exit loop if the solution pt found is closer to the next pt in path than the current pos
                    if pt_to_pt_distance(
                        goalPt, self.point_from_path(i + 1)
                    ) < pt_to_pt_distance(
                        [currentX, currentY], self.point_from_path(i + 1)
                    ):
                        print("got it, I'm out")
                        # update lastFoundIndex and exit
                        self.last_found_index = i
                        break
                    else:
                        print("moving to next index")
                        # in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
                        self.last_found_index = i + 1

                # if no solutions are in range
                else:
                    print("none in range")
                    foundIntersection = False
                    # no new intersection found, potentially deviated from the path
                    # follow path[lastFoundIndex]
                    goalPt = [
                        self.point_from_path(self.last_found_index)[0],
                        self.point_from_path(self.last_found_index)[1],
                    ]

            # if determinant < 0
            else:
                print("no intersection")
                foundIntersection = False
                # no new intersection found, potentially deviated from the path
                # follow path[lastFoundIndex]
                goalPt = [
                    self.point_from_path(self.last_found_index)[0],
                    self.point_from_path(self.last_found_index)[1],
                ]
        # else:
        #     print("probably nearing end? returning")
        #     return

        # obtained goal point, now compute turn vel
        # initialize proportional controller constant
        Kp = 0.8

        # calculate absTargetAngle with the atan2 function
        absTargetAngle = (
            math.atan2(
                goalPt[1] - self.odom.pose.pose.position.y,
                goalPt[0] - self.odom.pose.pose.position.x,
            )
            * 180
            / np.pi
        )
        if absTargetAngle < 0:
            absTargetAngle += 360

        # compute turn error by finding the minimum angle
        turnError = absTargetAngle - currentHeading
        if turnError > 180 or turnError < -180:
            turnError = -1 * sgn(turnError) * (360 - abs(turnError))
        print(
            "currentH, absAngle, err:",
            currentHeading,
            absTargetAngle,
            turnError,
            "-----------------",
        )

        # apply proportional controller
        turnVel = Kp * turnError

        # # experimental turnVel controller
        # W = 0.15
        # turnVel = 50 * W * math.sin(turnError) * linearVel / lookAheadDis

        # model: 200rpm drive with 18" width
        #               rpm   /s  circ   feet
        maxLinVel = 200 / 60 * np.pi  # * 4 / 12
        #               rpm   /s  center angle   deg
        maxTurnVel = 200 / 60 * np.pi * 4 / 9  # * 180 / np.pi

        final_linear_vel = linearVel / 100 * maxLinVel
        final_angular_vel = min(4, turnVel / 100 * maxTurnVel)

        print("linear, angular:", final_linear_vel, final_angular_vel)

        cmd_vel = Twist()
        cmd_vel.linear.x = float(final_linear_vel)
        cmd_vel.angular.z = float(final_angular_vel)
        self.cmd_vel_publisher.publish(cmd_vel)

        # return goalPt, lastFoundIndex, turnVel


def main(args=None):
    rclpy.init(args=args)

    pure_pursuit_controller = PurePursuit()
    rclpy.spin(pure_pursuit_controller)

    pure_pursuit_controller.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
