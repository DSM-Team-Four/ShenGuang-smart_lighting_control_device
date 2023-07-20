#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_gpio.h"
#include "iot_uart.h"
#include "hi_uart.h"
#include "iot_watchdog.h"
#include "iot_errno.h"
#include "iot_gpio_ex.h"

#define UART_BUFF_SIZE 15
#define U_SLEEP_TIME   100000

#define UART_TXD 0 
#define UART_RXD 1

void Uart1GpioInit(void);
void Uart1Config(void);
