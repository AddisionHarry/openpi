#include "communication/naviai_udp_protocal.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

// g++ -Iros/naviai_udp_comm/include     ros/naviai_udp_comm/src/communication/naviai_udp_protocal.cpp     ros/naviai_udp_comm/src/communication/test_protocal.cpp     -o test.o -lcrc32c -lpthread && ./test.o

using namespace NAVIAI_UDP_Communication;

static void test1(void)
{
    printf("***************************** test 1 *****************************\n");
    std::uint8_t data[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed = packData(data, 8, static_cast<Packet_Type_Enum>(0));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("packed pack number: %ld\n", packed.size());
    printf("packed pack[0] size: %ld, data : %s\n", packed.at(0).size(), vectorToHexString(packed.at(0)).c_str());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked = unpackData(packed.at(0));
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked.size());
    printf("unpacked data1: %s\n", vectorToHexString(unpacked.at(0).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test2(void)
{
    printf("***************************** test 2 *****************************\n");
    std::uint8_t data[1200] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed = packData(data, 1200, static_cast<Packet_Type_Enum>(0));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("packed pack number: %ld\n", packed.size());
    printf("packed pack[0] size: %ld, data : %s\n", packed.at(0).size(), vectorToHexString(packed.at(0)).c_str());
    printf("packed pack[1] size: %ld, data : %s\n", packed.at(1).size(), vectorToHexString(packed.at(1)).c_str());
    start = std::chrono::high_resolution_clock::now();
    unpackData(packed.at(0));
    auto unpacked = unpackData(packed.at(1));
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked.size());
    printf("unpacked data2 size: %ld, data: %s\n", unpacked.at(0).data.size(), vectorToHexString(unpacked.at(0).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test3(void)
{
    printf("***************************** test 3 *****************************\n");
    std::uint8_t data[1200] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed = packData(data, 1200, static_cast<Packet_Type_Enum>(1));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("packed pack number: %ld\n", packed.size());
    printf("packed pack[0] size: %ld, data : %s\n", packed.at(0).size(), vectorToHexString(packed.at(0)).c_str());
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < packed.size() - 1; ++i)
        unpackData(packed.at(i));
    auto unpacked = unpackData(packed.at(packed.size() - 1));
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked.size());
    printf("unpacked data3 size: %ld, data: %s\n", unpacked.at(0).data.size(), vectorToHexString(unpacked.at(0).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test4(void)
{
    printf("***************************** test 4 *****************************\n");
    std::uint8_t data[128] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed4 = packData(data, 64, static_cast<Packet_Type_Enum>(2));
    auto packed5 = packData(data, 128, static_cast<Packet_Type_Enum>(3));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::vector<std::uint8_t> combined_data;
    for (const auto &subvector : packed4)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    for (const auto &subvector : packed5)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked4 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked4.size());
    printf("unpacked data4[0] size: %ld, data: %s\n", unpacked4.at(0).data.size(), vectorToHexString(unpacked4.at(0).data).c_str());
    printf("unpacked data4[1] size: %ld, data: %s\n", unpacked4.at(1).data.size(), vectorToHexString(unpacked4.at(1).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test5(void)
{
    printf("***************************** test 5 *****************************\n");
    std::uint8_t data[512] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed6 = packData(data, 256, static_cast<Packet_Type_Enum>(2));
    auto packed7 = packData(data, 512, static_cast<Packet_Type_Enum>(3));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::vector<std::uint8_t> combined_data;
    combined_data.clear();
    for (const auto &subvector : packed6)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    for (const auto &subvector : packed7)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked5 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked5.size());
    printf("unpacked data5[0] size: %ld, data: %s\n", unpacked5.at(0).data.size(), vectorToHexString(unpacked5.at(0).data).c_str());
    printf("unpacked data5[1] size: %ld, data: %s\n", unpacked5.at(1).data.size(), vectorToHexString(unpacked5.at(1).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test6(void)
{
    printf("***************************** test 6 *****************************\n");
    std::vector<std::uint8_t> combined_data;
    std::vector<std::vector<std::uint8_t>> combined_data_tmp;
    std::uint8_t data[512] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed6 = packData(data, 256, static_cast<Packet_Type_Enum>(2));
    auto packed7 = packData(data, 512, static_cast<Packet_Type_Enum>(3));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed6.begin()), std::make_move_iterator(packed6.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed7.begin()), std::make_move_iterator(packed7.end()));
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(combined_data_tmp.begin(), combined_data_tmp.end(), g);
    combined_data.clear();
    for (const auto &subvector : combined_data_tmp)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked6 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked6.size());
    printf("unpacked data6[0] size: %ld, data: %s\n", unpacked6.at(0).data.size(), vectorToHexString(unpacked6.at(0).data).c_str());
    printf("unpacked data6[1] size: %ld, data: %s\n", unpacked6.at(1).data.size(), vectorToHexString(unpacked6.at(1).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test7(void)
{
    printf("***************************** test 7 *****************************\n");
    std::vector<std::uint8_t> combined_data;
    std::vector<std::vector<std::uint8_t>> combined_data_tmp;
    std::uint8_t data6[4096] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    auto start = std::chrono::high_resolution_clock::now();
    auto packed8 = packData(data6, 256, static_cast<Packet_Type_Enum>(2));
    auto packed9 = packData(data6, 512, static_cast<Packet_Type_Enum>(3));
    auto packed10 = packData(data6, 1024, static_cast<Packet_Type_Enum>(1));
    auto packed11 = packData(data6, 2048, static_cast<Packet_Type_Enum>(4));
    auto packed12 = packData(data6, 4096, Packet_Type_Enum::Test_Latency);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    combined_data_tmp.clear();
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed8.begin()), std::make_move_iterator(packed8.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed9.begin()), std::make_move_iterator(packed9.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed10.begin()), std::make_move_iterator(packed10.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed11.begin()), std::make_move_iterator(packed11.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed12.begin()), std::make_move_iterator(packed12.end()));
    combined_data.clear();
    for (const auto &subvector : combined_data_tmp)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked7 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked7.size());
    printf("unpacked data7[0] size: %ld, data: %s\n", unpacked7.at(0).data.size(), vectorToHexString(unpacked7.at(0).data).c_str());
    printf("unpacked data7[1] size: %ld, data: %s\n", unpacked7.at(1).data.size(), vectorToHexString(unpacked7.at(1).data).c_str());
    printf("unpacked data7[2] size: %ld, data: %s\n", unpacked7.at(2).data.size(), vectorToHexString(unpacked7.at(2).data).c_str());
    printf("unpacked data7[3] size: %ld, data: %s\n", unpacked7.at(3).data.size(), vectorToHexString(unpacked7.at(3).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test8(void)
{
    printf("***************************** test 8 *****************************\n");
    std::vector<std::uint8_t> combined_data;
    std::vector<std::vector<std::uint8_t>> combined_data_tmp;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(combined_data_tmp.begin(), combined_data_tmp.end(), g);
    combined_data.clear();
    for (const auto &subvector : combined_data_tmp)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    auto start = std::chrono::high_resolution_clock::now();
    auto unpacked8 = unpackData(combined_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked8.size());
    printf("unpacked data8[0] size: %ld, data: %s\n", unpacked8.at(0).data.size(), vectorToHexString(unpacked8.at(0).data).c_str());
    printf("unpacked data8[1] size: %ld, data: %s\n", unpacked8.at(1).data.size(), vectorToHexString(unpacked8.at(1).data).c_str());
    printf("unpacked data8[2] size: %ld, data: %s\n", unpacked8.at(2).data.size(), vectorToHexString(unpacked8.at(2).data).c_str());
    printf("unpacked data8[3] size: %ld, data: %s\n", unpacked8.at(3).data.size(), vectorToHexString(unpacked8.at(3).data).c_str());
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test9(void)
{
    printf("***************************** test 9 *****************************\n");
    std::uint8_t data7[20 * 1024] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    std::vector<std::uint8_t> combined_data;
    std::random_device rd;
    std::mt19937 g(rd());
    auto start = std::chrono::high_resolution_clock::now();
    auto packed13 = packData(data7, 20 * 1024, static_cast<Packet_Type_Enum>(2));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::shuffle(packed13.begin(), packed13.end(), g);
    combined_data.clear();
    for (const auto &subvector : packed13)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked9 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked9.size());
    printf("unpacked data9[0] size: %ld, data: %s\n", unpacked9.at(0).data.size(), vectorToHexString(unpacked9.at(0).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test10(void)
{
    printf("***************************** test 10 *****************************\n");
    std::uint8_t data8[8192] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    std::vector<std::uint8_t> combined_data;
    std::vector<std::vector<std::uint8_t>> combined_data_tmp;
    std::random_device rd;
    std::mt19937 g(rd());
    auto start = std::chrono::high_resolution_clock::now();
    auto packed14 = packData(data8, 1024, static_cast<Packet_Type_Enum>(2));
    auto packed15 = packData(data8, 2048, static_cast<Packet_Type_Enum>(3));
    auto packed16 = packData(data8, 4096, static_cast<Packet_Type_Enum>(1));
    auto packed17 = packData(data8, 4096, static_cast<Packet_Type_Enum>(4));
    auto packed18 = packData(data8, 8192, Packet_Type_Enum::Test_Latency);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    combined_data_tmp.clear();
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed14.begin()), std::make_move_iterator(packed14.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed15.begin()), std::make_move_iterator(packed15.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed16.begin()), std::make_move_iterator(packed16.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed17.begin()), std::make_move_iterator(packed17.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed18.begin()), std::make_move_iterator(packed18.end()));
    std::shuffle(combined_data_tmp.begin(), combined_data_tmp.end(), g);
    combined_data.clear();
    for (const auto &subvector : combined_data_tmp)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    start = std::chrono::high_resolution_clock::now();
    auto unpacked10 = unpackData(combined_data);
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("unpacked packs: %ld\n", unpacked10.size());
    printf("unpacked data10[0] size: %ld, data: %s\n", unpacked10.at(0).data.size(), vectorToHexString(unpacked10.at(0).data).c_str());
    printf("unpacked data10[1] size: %ld, data: %s\n", unpacked10.at(1).data.size(), vectorToHexString(unpacked10.at(1).data).c_str());
    printf("unpacked data10[2] size: %ld, data: %s\n", unpacked10.at(2).data.size(), vectorToHexString(unpacked10.at(2).data).c_str());
    printf("unpacked data10[3] size: %ld, data: %s\n", unpacked10.at(3).data.size(), vectorToHexString(unpacked10.at(3).data).c_str());
    std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
    std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
}

static void test11(void)
{
    printf("***************************** test 11 *****************************\n");
    std::uint8_t data8[8192] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    std::vector<std::uint8_t> combined_data;
    std::vector<std::vector<std::uint8_t>> combined_data_tmp;
    std::random_device rd;
    std::mt19937 g(rd());
    auto start = std::chrono::high_resolution_clock::now();
    auto packed14 = packData(data8, 1024, static_cast<Packet_Type_Enum>(2));
    auto packed15 = packData(data8, 2048, static_cast<Packet_Type_Enum>(3));
    auto packed16 = packData(data8, 4096, static_cast<Packet_Type_Enum>(1));
    auto packed17 = packData(data8, 4096, static_cast<Packet_Type_Enum>(4));
    auto packed18 = packData(data8, 8192, Packet_Type_Enum::Test_Latency);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    combined_data_tmp.clear();
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed14.begin()), std::make_move_iterator(packed14.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed15.begin()), std::make_move_iterator(packed15.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed16.begin()), std::make_move_iterator(packed16.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed17.begin()), std::make_move_iterator(packed17.end()));
    combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed18.begin()), std::make_move_iterator(packed18.end()));
    std::shuffle(combined_data_tmp.begin(), combined_data_tmp.end(), g);
    combined_data.clear();
    for (const auto &subvector : combined_data_tmp)
        combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
    printf("packed pack size %ld bytes.\n", combined_data.size());
    std::vector<long long> durations;
    std::vector<size_t> unpacked_sizes;
    for (size_t i = 0; i < 20000; ++i)
    {
        start = std::chrono::high_resolution_clock::now();
        auto unpacked11 = unpackData(combined_data);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        durations.push_back(duration2.count());
        unpacked_sizes.push_back(unpacked11.size());
        if (i % 1024 == 0)
            std::cout << i << " / 20000" << std::endl;
    }
    auto write_to_csv = [](const std::vector<long long> &durations, const std::vector<size_t> &unpacked_sizes,
                           std::string &&file_name) -> void {
        std::ofstream file(file_name);
        if (file.is_open())
        {
            file << "Duration (ns), Unpacked Size\n";
            for (size_t i = 0; i < durations.size(); ++i)
                file << durations[i] << ", " << unpacked_sizes[i] << "\n";
            file.close();
            std::cout << "Data saved to " << file_name << std::endl;
        }
        else
            std::cerr << "Unable to open file for writing." << std::endl;
    };
    write_to_csv(durations, unpacked_sizes, "/tmp/test_udp_protocals_test11.csv");
}

static void test12(void)
{
    printf("***************************** test 12 *****************************\n");
    auto read_jpg_file = [](const std::string &filename, std::vector<std::uint8_t> &data) -> bool {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(file_size);
        file.read(reinterpret_cast<char *>(data.data()), file_size);
        if (!file)
        {
            std::cerr << "Error reading file: " << filename << std::endl;
            return false;
        }
        return true;
    };
    auto save_jpg_from_vector = [](const std::vector<std::uint8_t> &image_data, const std::string file_path) -> void {
        std::ofstream output_file(file_path, std::ios::binary);
        if (!output_file)
        {
            std::cerr << "Failed to open output file!" << std::endl;
            return;
        }
        output_file.write(reinterpret_cast<const char *>(image_data.data()), image_data.size());
        if (output_file)
            std::cout << "Image successfully saved to: " << file_path << std::endl;
        else
            std::cerr << "Failed to write image data to file!" << std::endl;
    };
    std::vector<std::uint8_t> data;
    if (read_jpg_file("/tmp/test1.jpg", data))
    {
        std::cout << "Successfully read " << data.size() << " bytes from the file." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto packed1 = packData(data.data(), data.size(), static_cast<Packet_Type_Enum>(1));
        auto packed2 = packData(data.data(), data.size(), static_cast<Packet_Type_Enum>(2));
        auto packed3 = packData(data.data(), data.size(), static_cast<Packet_Type_Enum>(3));
        auto packed4 = packData(data.data(), data.size(), static_cast<Packet_Type_Enum>(4));
        auto packed5 = packData(data.data(), data.size(), static_cast<Packet_Type_Enum>(0));
        // auto packed6 = packData(data.data(), data.size(), Packet_Type_Enum::Center_Image_Feedback);
        // auto packed7 = packData(data.data(), data.size(), Packet_Type_Enum::Center_Image_Feedback);
        // auto packed8 = packData(data.data(), data.size(), Packet_Type_Enum::Center_Image_Feedback);
        // auto packed9 = packData(data.data(), data.size(), Packet_Type_Enum::Center_Image_Feedback);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        std::random_device rd;
        std::mt19937 g(rd());
        std::vector<std::vector<std::uint8_t>> combined_data_tmp;
        combined_data_tmp.clear();
        combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed1.begin()), std::make_move_iterator(packed1.end()));
        combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed2.begin()), std::make_move_iterator(packed2.end()));
        combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed3.begin()), std::make_move_iterator(packed3.end()));
        combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed4.begin()), std::make_move_iterator(packed4.end()));
        combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed5.begin()), std::make_move_iterator(packed5.end()));
        // combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed6.begin()), std::make_move_iterator(packed6.end()));
        // combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed7.begin()), std::make_move_iterator(packed7.end()));
        // combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed8.begin()), std::make_move_iterator(packed8.end()));
        // combined_data_tmp.insert(combined_data_tmp.end(), std::make_move_iterator(packed9.begin()), std::make_move_iterator(packed9.end()));
        std::shuffle(combined_data_tmp.begin(), combined_data_tmp.end(), g);
        std::vector<std::uint8_t> combined_data;
        for (const auto &subvector : combined_data_tmp)
            combined_data.insert(combined_data.end(), subvector.begin(), subvector.end());
        start = std::chrono::high_resolution_clock::now();
        auto unpacked = unpackData(combined_data);
        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        printf("unpacked packs: %ld\n", unpacked.size());
        std::cout << "Pack Time: " << duration1.count() << " nanoseconds." << std::endl;
        std::cout << "UnPack Time: " << duration2.count() << " nanoseconds." << std::endl;
        save_jpg_from_vector(unpacked.at(0).data, "/tmp/test2.jpg");
    }
    else
        std::cerr << "Failed to read the file." << std::endl;
}

// Test Code
int __attribute__((weak)) main(int argc, char **argv)
{
    test2();
    return 0;
}
