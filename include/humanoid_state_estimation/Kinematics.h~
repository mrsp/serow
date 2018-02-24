/*
 * hrl_kinematics - a kinematics library for humanoid robots based on KDL
 *
 * Copyright 2011-2012 Armin Hornung, University of Freiburg
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HRL_KINEMATICS_KINEMATICS_H_
#define HRL_KINEMATICS_KINEMATICS_H_

#include <string>
#include <map>
#include <exception>
#include <math.h>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_kdl.h>
#include <tf/transform_datatypes.h>
#include <urdf/model.h>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/frames.hpp>
#include <kdl/segment.hpp>
#include <kdl/joint.hpp>

#include <robot_state_publisher/robot_state_publisher.h>
#include <kdl_parser/kdl_parser.hpp>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/MarkerArray.h>


namespace hrl_kinematics {
typedef std::map<std::string, double> JointMap;

/**
 * Class to compute the center of mass while recursively traversing
 * a kinematic tree of a robot (from URDF)
 * 
 */
class Kinematics {
public:
  enum FootSupport {SUPPORT_DOUBLE, SUPPORT_SINGLE_RIGHT, SUPPORT_SINGLE_LEFT};
  class InitFailed : public std::runtime_error
  {
  public:
    InitFailed(const std::string &what)
    : std::runtime_error(what) {}
  };

  /**
   * \brief Constructor
   * \param root_link_name - name of link that is base of robot
   * \param rfoot_link_name - name of link that is considered the right foot
   * \param lfoot_link_name - name of link that is considered the left foot
   * \param urdf_model - a pointer to a pre-loaded URDF model that can speed up initialization if provided
   */
  Kinematics(std::string root_link_name = "base_link", std::string rfoot_link_name = "r_sole", std::string lfoot_link_name = "l_sole",
             const boost::shared_ptr<const urdf::ModelInterface>& urdf_model = boost::shared_ptr<const urdf::ModelInterface>());

  virtual ~Kinematics();
  void initialize();

  /**
   * Computes the center of mass of the given robot structure and joint configuration.
   * Will also return the 6D transformations to the feet while traversing the robot tree
   *
   * @param[in] joint_positions angles of all joints
   * @param[out] com the computed center of mass in the root coordinate frame (base_link)
   * @param[out] mass total mass of all joints
   * @param[out] tf_right_foot returned transformation from body reference frame to right foot
   * @param[out] tf_left_foot returned transformation from body reference frame to left foot
   */
  void computeCOM(const JointMap& joint_positions, tf::Point& com, double& mass,
                  tf::Transform& tf_right_foot, tf::Transform& tf_left_foot);


  protected:
  bool loadKDLModel();
  void addChildren(const KDL::SegmentMap::const_iterator segment);

  void computeCOMRecurs(const KDL::SegmentMap::const_iterator& currentSeg, const std::map<std::string, double>& joint_positions,
                        const KDL::Frame& tf, KDL::Frame& tf_right_foot, KDL::Frame& tf_left_foot, double& m, KDL::Vector& cog);
  void createCoGMarker(const std::string& ns, const std::string& frame_id, double radius, const KDL::Vector& cog, visualization_msgs::Marker& marker) const;

  boost::shared_ptr<const urdf::ModelInterface> urdf_model_;
  KDL::Tree kdl_tree_;
  KDL::Chain kdl_chain_right_;
  KDL::Chain kdl_chain_left_;

  ros::NodeHandle nh_, nh_private_;
  std::string root_link_name_;
  std::string rfoot_link_name_;
  std::string lfoot_link_name_;

  unsigned int num_joints_;
  std::map<std::string, robot_state_publisher::SegmentPair> segments_;
};

} /* namespace hrl_kinematics */
#endif
