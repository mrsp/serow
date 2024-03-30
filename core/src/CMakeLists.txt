find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)

set(serow_src ${serow_src} 
    Serow.cpp 
    State.cpp 
    ContactEKF.cpp 
    CoMEKF.cpp 
    LegOdometry.cpp 
    ButterworthLPF.cpp 
    Differentiator.cpp 
    JointEstimator.cpp 
    DerivativeEstimator.cpp)

set(serow_headers ${serow_headers} 
    Serow.hpp 
    State.hpp 
    ContactEKF.hpp 
    CoMEKF.hpp 
    LegOdometry.hpp 
    ButterworthLPF.hpp 
    Differentiator.hpp 
    JointEstimator.hpp 
    DerivativeEstimator.hpp
    ContactDetector.hpp
    lie.hpp
    Mahony.hpp
    Measurement.hpp
    MovingMedianFilter.hpp
    RobotKinematics.hpp)

add_library(serow SHARED ${serow_headers} ${serow_src})
target_link_libraries(serow PRIVATE Eigen3::Eigen pinocchio::pinocchio nlohmann_json::nlohmann_json)

install(FILES ${serow_headers} DESTINATION include/serow)
install(TARGETS serow DESTINATION lib/serow)
install(FILES serowConfig.cmake DESTINATION lib/cmake/serow)
