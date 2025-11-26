#include "communication/udp_comm.h"

#include <arpa/inet.h> // for sockaddr_in, inet_addr
#include <atomic>      // for atomic flags
#include <condition_variable>
#include <cstring> // for memset
#include <iostream>
#include <mutex> // for thread safety
#include <queue>
#include <thread>
#include <unistd.h>      // for close
#include <unordered_map> // for multiple ports and callbacks
#include <vector>

namespace NAVIAI_UDP_Communication
{
#define RECV_BUFFER_SIZE 1024
#define RECV_BUFFER_NUMBER 3

class ThreadPool
{
  public:
    ThreadPool(size_t numThreads)
    {
        for (size_t i = 0; i < numThreads; ++i)
            _threads.emplace_back(&ThreadPool::worker, this);
    }

    virtual ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            _stop = true;
        }
        _condVar.notify_all();
        for (auto &thread : _threads)
            if (thread.joinable())
                thread.join();
    }

    void enqueueTask(const std::function<void(void)> &task)
    {
        {
            std::unique_lock<std::mutex> lock(_queueMutex);
            if (_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            _tasks.push(task);
        }
        _condVar.notify_one();
    }

  private:
    void worker(void)
    {
        while (true)
        {
            std::function<void(void)> task;
            {
                std::unique_lock<std::mutex> lock(_queueMutex);
                _condVar.wait(lock, [this]() { return (!_tasks.empty() || _stop); });
                if (_stop && _tasks.empty())
                    return;
                task = _tasks.front();
                _tasks.pop();
            }
            task();
        }
    }

    std::vector<std::thread> _threads;
    std::queue<std::function<void(void)>> _tasks;
    std::mutex _queueMutex;
    std::condition_variable _condVar;
    bool _stop = false;
};

static bool _isRunning = true;
static ThreadPool threadPool(4);

static size_t _use_recv_buffer_index = 0;
static std::unordered_map<int, std::thread> _receiver_threads;
static std::unordered_map<int, UDP_Recv_Callback> _recv_callbacks;
static std::mutex _recv_callback_mutex;
static std::uint8_t _recv_buffer[RECV_BUFFER_NUMBER][RECV_BUFFER_SIZE];

static int _send_sockfd = -1;
static sockaddr_in _sender_server_addr;

static void _recv_thread(int sockfd, int port)
{
    sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    std::cout << "Listening on port " << port << "..." << std::endl;
    while (_isRunning)
    {
        std::uint8_t *buffer = _recv_buffer[_use_recv_buffer_index];
        int n = recvfrom(sockfd, buffer, RECV_BUFFER_SIZE, 0, (sockaddr *)&client_addr, &addr_len);
        if (n > 0)
        {
            std::string sender_ip = inet_ntoa(client_addr.sin_addr);
            int sender_port = ntohs(client_addr.sin_port);
            UDP_Recv_Callback use_callback = nullptr;
            {
                std::lock_guard<std::mutex> lock(_recv_callback_mutex);
                if (_recv_callbacks.find(port) != _recv_callbacks.end())
                    use_callback = _recv_callbacks[port];
            }
            if (use_callback)
            {
                threadPool.enqueueTask(
                    [use_callback, buffer, n, sender_port, &sender_ip](void) -> void { use_callback(buffer, n, sender_ip, sender_port); });
            }
        }
        _use_recv_buffer_index = (_use_recv_buffer_index + 1) % RECV_BUFFER_NUMBER;
    }
    std::cout << "Receiver on port " << port << " stopped." << std::endl;
    close(sockfd);
}

bool init_udp_receiver(int port)
{
    int sockfd;
    sockaddr_in server_addr;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        std::cerr << "Socket creation failed for port " << port << std::endl;
        return false;
    }
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);
    if (bind(sockfd, (const sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        std::cerr << "Bind failed for port " << port << std::endl;
        close(sockfd);
        return false;
    }
    std::thread([sockfd, port]() -> void { _recv_thread(sockfd, port); }).detach();
    return true;
}

void udp_receive_register_callback(int port, UDP_Recv_Callback callback)
{
    std::lock_guard<std::mutex> lock(_recv_callback_mutex);
    _recv_callbacks[port] = callback;
}

void udp_receive_stop_receiver(int port)
{
    std::lock_guard<std::mutex> lock(_recv_callback_mutex);
    if (_recv_callbacks.find(port) != _recv_callbacks.end())
        _recv_callbacks.erase(port);
}

void udp_receive_stop_all_receivers(void)
{
    _isRunning = false;
    _recv_callbacks.clear();
}

bool init_udp_sender(const std::string &target_ip, int target_port)
{
    if (_send_sockfd != -1)
    {
        std::cerr << "Sender is already initialized." << std::endl;
        return false;
    }
    _send_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (_send_sockfd < 0)
    {
        std::cerr << "Socket creation failed." << std::endl;
        return false;
    }
    memset(&_sender_server_addr, 0, sizeof(_sender_server_addr));
    _sender_server_addr.sin_family = AF_INET;
    _sender_server_addr.sin_port = htons(target_port);
    _sender_server_addr.sin_addr.s_addr = inet_addr(target_ip.c_str());
    return true;
}

ssize_t udp_sender_send(const std::uint8_t *message, size_t data_length)
{
    if (_send_sockfd == -1)
    {
        std::cerr << "Sender is not initialized. Call udp_sender_initialize() first." << std::endl;
        return 0;
    }
    ssize_t res = sendto(_send_sockfd, message, data_length, 0, (const sockaddr *)&_sender_server_addr, sizeof(_sender_server_addr));
    if (res < 0)
        std::cerr << "Send failed to " << inet_ntoa(_sender_server_addr.sin_addr) << ":" << ntohs(_sender_server_addr.sin_port)
                  << std::endl;
    return res;
}

void udp_sender_shutdown(void)
{
    if (_send_sockfd != -1)
    {
        close(_send_sockfd);
        _send_sockfd = -1;
        std::cout << "Sender connection closed." << std::endl;
    }
}
} // namespace NAVIAI_UDP_Communication
