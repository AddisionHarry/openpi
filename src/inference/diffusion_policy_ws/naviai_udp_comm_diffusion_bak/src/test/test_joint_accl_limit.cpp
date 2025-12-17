#include "accl_limit/joint_accl_limit.h"

#include <fstream>
#include <iomanip>
#include <iostream>

// g++ ros/naviai_udp_comm/src/test_joint_accl_limit.cpp ros/naviai_udp_comm/src/joint_accl_limit.* -o test_accl_limit.o --std=c++17 && ./test_accl_limit.o

static void saveStatesToCSV(const std::vector<std::vector<float>> &targetStates, const std::vector<std::vector<float>> &currentStates,
                            const std::string filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file " << filename << std::endl;
        return;
    }
    file << "target_position,target_velocity,current_position,current_velocity\n";
    // Write the data rows
    for (size_t i = 0; i < targetStates.size(); ++i)
    {
        // targetStates[i] contains target.position and target.velocity
        // currentStates[i] contains current.position and current.velocity
        file << std::fixed << std::setprecision(6) << targetStates[i][0] << "," // target.position
             << targetStates[i][1] << ","                                       // target.velocity
             << currentStates[i][0] << ","                                      // current.position
             << currentStates[i][1]                                             // current.velocity
             << "\n";
    }
    // Close the file
    file.close();
}

int __attribute__((weak)) main(int argc, char **argv)
{
    auto target_position = [](float t) -> float { return -2 * t + 1 + 5 * std::sin(6 * t) * std::exp(-0.1 * t) + 3 * std::abs(t - 10.5); };
    DynamicsProtecter1D protector(45, -45, 300);
    DynamicsState current(120, 10), target(0, 0);
    std::vector<std::vector<float>> targetStates, currentStates;
#define SIMULATE_TIME_LENGTH 50000
    targetStates.resize(SIMULATE_TIME_LENGTH);
    currentStates.resize(SIMULATE_TIME_LENGTH);
    for (int i = 0; i < SIMULATE_TIME_LENGTH; ++i)
    {
        std::cout << i << std::endl;
        float time = static_cast<float>(i) * 0.001;
        target.position = target_position(time);
        target.velocity = (target_position(time + 0.001) - target_position(time - 0.001)) / 0.002;
        targetStates.at(i).reserve(2);
        targetStates.at(i).push_back(target.position);
        targetStates.at(i).push_back(target.velocity);
        currentStates.at(i).reserve(2);
        currentStates.at(i).push_back(current.position);
        currentStates.at(i).push_back(current.velocity);
        current = protector.calculateAcclClamp(current, target, 0.001);
        if (std::abs(current.position - currentStates.at(i).at(0)) > 1.5)
        {
            std::cout << "Get interrupt at current state: position: " << currentStates.at(i).at(0)
                      << ", velocity: " << currentStates.at(i).at(1) << ", target state: position: " << targetStates.at(i).at(0)
                      << ", velocity: " << targetStates.at(i).at(1) << ". Get new target: position: " << current.position
                      << ", velocity: " << current.velocity << std::endl;
            std::ofstream null_stream("/dev/null");
            std::cout.rdbuf(null_stream.rdbuf());
        }
    }
    saveStatesToCSV(targetStates, currentStates, "/tmp/test_planning.csv");
    return 0;
}
