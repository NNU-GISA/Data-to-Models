# Data-to-Models
Thesis project repository. Will be used to store mostly c++ code related to the transformation of raw point cloud data to solid models of structural components.

<b>Current progress:</b>

Environment for reading raw LIDAR rosbag data with BLAM is <i>mostly</i> functional.

<b>Next steps:</b>

Export the registered point cloud data as one or multiple pcd files.

Manually or automatically clean point clouds to remove outliers and non-relevant points from the data set

Impliment the algorithm described in Lu paper to segment point cloud into multiple sub clouds each containing a labeled structural component (pier, decking, girder, pier cap, column, ect)

Spline these component point cloud datasets to create solid models. Consider using a system as described in the Nao paper for this and/or the previous step as well.

Combine resultant solid models into a single entity based on relative position from original data.

Create finite element model from this combined model. Estimate material and connectivity parameters by some TBD algorithm.

Using in-situ testing of actual bridge under static and/or dynamic loading, refine the material and connectivity parameters through a TBD machine-learning type iterative approach.
