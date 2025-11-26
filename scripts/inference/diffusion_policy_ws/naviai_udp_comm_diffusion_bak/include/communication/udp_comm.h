#ifndef UDP_COMM_H
#define UDP_COMM_H

#include <cstddef>
#include <functional>
#include <string>

namespace NAVIAI_UDP_Communication
{
using UDP_Recv_Callback = std::function<void(const std::uint8_t *data, size_t data_length, const std::string &sender_ip, int sender_port)>;

extern bool init_udp_sender(const std::string &target_ip, int target_port);
extern ssize_t udp_sender_send(const std::uint8_t *message, size_t data_length);
extern void udp_sender_shutdown(void);

extern bool init_udp_receiver(int port);
extern void udp_receive_register_callback(int port, UDP_Recv_Callback callback);
extern void udp_receive_stop_receiver(int port);
extern void udp_receive_stop_all_receivers(void);
}; // namespace NAVIAI_UDP_Communication

#endif /* UDP_COMM_H */
