import re
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

"""

This starter code provides code base to implement point to plane ICP for HW5a Q4a
This is a stand alone code and does not require ROS
--- TBD --- indicates the section to fill out   
... indicates the specific place to type the code

"""


class DataParser:
    """Parses odometry and scan data from log files.
    This class provides functionality to read and parse ROS2 bag file2txt dumps
    NO ***** TBD ********* items in this class
    """

    def __init__(self, odom_path, scan_path):
        self.odom_path = odom_path
        self.scan_path = scan_path
        self.odom_data = []
        self.scan_data = []
        self.x = []
        self.y = []
        self.theta = []

    def quaternion_to_yaw(self, z, w):
        return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)

    def parse_odom(self):
        results = []
        x = []
        y = []
        yaw = []
        timestamp = []
        with open(self.odom_path, 'r') as f:
            timestamp_ = None
            for line in f:
                line = line.strip()
                if line.startswith("# Time:"):
                    timestamp_ = float(line.split(":")[1].strip())
                elif line.startswith("nav_msgs.msg.Odometry("):
                    pos_match = re.search(r"position=geometry_msgs\.msg\.Point\(x=([-\de.E]+), y=([-\de.E]+)", line)
                    ori_match = re.search(
                        r"orientation=geometry_msgs\.msg\.Quaternion\(x=[-\de.E]+, y=[-\de.E]+, z=([-\de.E]+), w=([-\de.E]+)",
                        line)
                    if pos_match and ori_match and timestamp_ is not None:
                        x_ = float(pos_match.group(1))
                        y_ = float(pos_match.group(2))
                        z_ = float(ori_match.group(1))
                        w_ = float(ori_match.group(2))
                        yaw_ = self.quaternion_to_yaw(z_, w_)
                        results.append((timestamp_, x_, y_, yaw_))
                        x.append(x_)
                        y.append(y_)
                        yaw.append(yaw_)
                        timestamp.append(timestamp_)
                        timestamp_ = None
        self.odom_data = np.vstack(results)
        self.x = np.array(x)
        self.y = np.array(y)
        self.theta = np.array(yaw)
        print("No. of odom parsed: ", self.theta.shape)

    def parse_scan(self):
        with open(self.scan_path, 'r') as f:
            text = f.read()
        entries = text.split("# Time:")
        angle_min = []
        angle_max = []
        angle_increment = []
        ranges = []
        for entry in entries[1:]:
            time_match = re.search(r'(\d+\.\d+)', entry)
            scan_match = re.search(
                r'angle_min=([\-\d.e]+), angle_max=([\-\d.e]+), angle_increment=([\-\d.e]+),.*?ranges=\[(.*?)\]', entry,
                re.DOTALL)
            if time_match and scan_match:
                angle_min_ = float(scan_match.group(1))
                angle_max_ = float(scan_match.group(2))
                angle_increment_ = float(scan_match.group(3))
                ranges_str = scan_match.group(4).replace('\n', '').replace(' ', '')
                ranges_ = [float(r) if r != 'nan' else np.nan for r in ranges_str.split(',') if r]
                angle_min.append(angle_min_)
                angle_max.append(angle_max_)
                angle_increment.append(angle_increment_)
                ranges.append(ranges_)
        print("No. of lidar scan parsed: ", np.asarray(angle_increment).shape)
        self.scan_data = (ranges, angle_min, angle_max, angle_increment)


class Point2PlaneICP:
    def __init__(self, parser, skip_pose=10):
        self.parser = parser
        self.skip_pose = skip_pose
        self.accumulated_points = []
        self.visualize_normal = False  # set this to true to visualize the normals
        self.normal_simple = True  # use this to switch between a simple and more robust normal estimation

    def ranges_to_xy(self, ranges, angles):
        """
        Polar coordinate to cartesian conversion. Borrowed from Hw4
        """
        ranges = np.array(ranges)
        angles = angles[:len(ranges)]
        valid = ~np.isnan(ranges)
        return np.vstack([
            ranges[valid] * np.cos(angles[valid]),
            ranges[valid] * np.sin(angles[valid])
        ]).T

    def transform_points(self, points, pose):
        """
        Odomtery based point cloud merge. here used for good initialization. Borrowed from HW4
        """
        x, y, theta = pose
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        t = np.array([x, y])
        return (R @ points.T).T + t

    def compute_normals(self, points, sensor_origin):

        # --------------- TBD ---------------
        #************************************************
        #           IMPORTANT DETAILS BELOW             #
        #************************************************
        """
        Three point normals. i.e compute normals using 3 neighbors
        This is the simplest form of normals.
        A more sophisticated version is available under compute_normals_pca()
        using Principle Component Analysis
            -Use this function to check if you normals looks similar
            - To use compute_normals_pca(), set self.normal_simple = False

        - for point cloud index from 0 to n,
        - step 1: start from index 1 to n-1, i.e leaving the first and last point
        - step 2: in point clouds, normals = previous point - next point
            - this gives you a central difference approximation of the tangent
            - then rotate 90 deg to get the normal
            - e.g. tangent = p[i + 1] - p[i - 1] = [4, 5] - [2, 3] = [2, 2]
            - step 3: flip this tangent to get the normal. i.e. normal = [-tangent[1], tangent[0]] = [-2, 2]
        - step 4: find the normal = normal/np.linalg.norm(normal)
        - step 5: the sensor_origin is used to align normals in the same directions
        - step 6: append to the list normals[]
        - step 7: normals[0] = normals[1], and normals[n] = normals[n-1]
        - visualize the normals using the function plot_normals()
            i.e enable self.visualize_normal under init()
         """
        normals = []
        for i in range(1, len(points) - 1):
            p1, p2 = points[i - 1], points[i + 1]
            tangent = p2 - p1
            normal = [-tangent[1], tangent[0]]
            normal = normal/np.linalg.norm(normal)
            to_sensor = sensor_origin - points[i]
            if np.dot(normal, to_sensor) < 0:
                normal = -normal
            normals.append(normal)
        normals.insert(0, normals[1]) 
        normals.append(normals[-1]) 
        return np.array(normals)

    def compute_normals_pca(self, points, tree, sensor_origin, k=10):
        """
        More sophisticated way to compute normals. Here using PCA
        This was not explicitly covered in class but similar to the concept of SVD principal directions
        Use this function to visualize normals and compare it to your method
        """
        normals = []
        for pt in points:
            _, idxs = tree.query(pt, k=k)
            neighbors = points[idxs]
            cov = np.cov(neighbors.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, 0]
            normal /= np.linalg.norm(normal)
            to_sensor = sensor_origin - pt
            if np.dot(normal, to_sensor) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)

    def icp_point_to_plane(self, source_points, target_points, target_sensor_origin, max_iterations=15):

        """
        This is the main function that computes the point to plane ICP

        """
        src = np.copy(source_points)
        tgt = np.copy(target_points)
        tree = cKDTree(tgt)

        if self.normal_simple:  # a mechanism to switc between your normals and the robust normals estimates
            normals = self.compute_normals(tgt, target_sensor_origin)
        else:
            normals = self.compute_normals_pca(tgt, tree, target_sensor_origin, k=8)

        if self.visualize_normal:
            self.plot_normals(tgt, normals)


        # ----------------------- TBD -------------------
        for _ in range(max_iterations):
            _, indices = tree.query(src)  # use kd tree for initial association.
            matched_pts = tgt[indices]    # get the matched points from target PCL using the kd tree index
            matched_normals = normals[indices]  # get the matched normals using the kd tree index
            A, b = [], []
            for p, q, n in zip(src, matched_pts, matched_normals):
                # using δ, n (normals) form the parts of the linear system i.e. A and b as the system is Ax=b
                # refer to the lecture slides for more details on the equations
                # A = [n Rp n] is the matrix form, use the linearized form for the Rp
                #   i.e. A = [n · [-py, px], nx, ny]
                # b = [-n δ] =>  b = -n · (p - q)
                # when working with vectors, use dot product where applicable
                # Append A and b to the list

                delta = p - q
                px, py = p
                nx, ny = n

                A.append([np.dot(n, [-py, px]), nx, ny])  # append [n · [-py, px], nx, ny]
                b.append(np.dot(-n, delta))  # append b = -n · (p - q)

            A = np.array(A)
            b = np.array(b)

            # use the np.linalg.lstsq() to solve for the least square.
            #    This function performs SVD in the background but is more stable to conventional SVD.
            # extract the angle, tx, ty from the above step
            # form a rotation matrix, and a translation vector
            # transform the source using the R and t from above
            #    alternatively form a homogeneous transformation matrix and transform the source
            # return source point cloud

            x, _, _, _ = np.linalg.lstsq(A, b) # use np.np.linalg.lstsq() to solve SVD
            angle = x[0]
            tx = x[1]
            ty = x[2]
            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) # rotation matrix
            # update src using the R and t or the homogeneous matrix
            src = (R @ src.T).T + np.array([tx, ty])
            # ----------------------- TBD-END -------------------
        return src

    def plot_normals(self, points, normals, title="Point Normals"):
        """
        plot the normals
        """
        points = np.array(points)
        normals = np.array(normals)
        plt.figure(figsize=(8, 8))
        plt.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1],
                   angles='xy', scale_units='xy', scale=15, color='r', width=0.003, label='Normals')
        plt.plot(points[:, 0], points[:, 1], 'b.', markersize=2, label='Points')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def visualize_odom(self):
        """
        diplay's the robot's movement while collecting the laser scans
        """
        x, y, theta = self.parser.x, self.parser.y, self.parser.theta
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o', markersize=2, linestyle='-', label="Odometry Path")
        dx = np.cos(theta)
        dy = np.sin(theta)
        plt.quiver(x[::10], y[::10], dx[::10], dy[::10], angles='xy', scale_units='xy', scale=15, color='r',
                   width=0.005, label='Orientation')
        plt.title("Odometry Trajectory with Orientation")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run_icp(self):
        """
        This is the function that reads the data and runs icp in loop

        """
        accumulated_pc = None
        ranges, angle_mins, angle_maxs, _ = self.parser.scan_data
        x = self.parser.x
        y = self.parser.y
        theta = self.parser.theta
        for i in range(0, len(ranges), self.skip_pose):
            r = ranges[i]
            angles = np.linspace(angle_mins[i], angle_maxs[i], len(r))
            local_pts = self.ranges_to_xy(r, angles)
            pose = (x[i], y[i], theta[i])
            initialized_pts = self.transform_points(local_pts, pose)  # this step makes it esier for icp to converge
            sensor_origin = np.array([pose[0], pose[1]])
            if accumulated_pc is None:
                accumulated_pc = initialized_pts
                self.accumulated_points.append(initialized_pts)
            else:
                aligned = self.icp_point_to_plane(initialized_pts, accumulated_pc, sensor_origin)
                accumulated_pc = np.vstack((accumulated_pc, aligned))
                self.accumulated_points.append(aligned)
            print(f"Progress: {np.round(i / len(ranges) * 100.0, 2)}%", end='\r')

    def plot_map(self):
        plt.figure(figsize=(8, 8))
        for pc in self.accumulated_points:
            plt.scatter(pc[:, 0], pc[:, 1], s=1)
        plt.title("Accumulated Point Cloud via Point-to-Plane ICP")
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    path_ = "/Users/damianmolina/Downloads/16280-hw5/hw5_pkg/data/" # change this to match your directory. note: '/' expected at the end
    parser = DataParser(odom_path=path_ + "odom_sync.txt", scan_path=path_ + "scan_sync.txt")
    parser.parse_odom()
    parser.parse_scan()
    mapper = Point2PlaneICP(parser=parser, skip_pose=5)
    mapper.visualize_odom()
    mapper.run_icp()
    mapper.plot_map()
