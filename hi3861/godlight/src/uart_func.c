/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "uart_func.h"

void Uart1GpioInit(void)
{
    IoTGpioInit(UART_TXD);
    // 设置GPIO0的管脚复用关系为UART1_TX Set the pin reuse relationship of GPIO0 to UART1_ TX
    IoTGpioSetFunc(UART_TXD, IOT_GPIO_FUNC_GPIO_0_UART1_TXD);

    IoTGpioInit(UART_RXD);
    // 设置GPIO1的管脚复用关系为UART1_RX Set the pin reuse relationship of GPIO1 to UART1_ RX
    IoTGpioSetFunc(UART_RXD, IOT_GPIO_FUNC_GPIO_1_UART1_RXD);
}

void Uart1Config(void)
{
    uint32_t ret;
    /* 初始化UART配置，波特率 115200，数据bit�??8,停止�??1，奇偶校验为NONE */
    /* Initialize UART configuration, baud rate is 9600, data bit is 8, stop bit is 1, parity is NONE */
    IotUartAttribute uart_attr = {
        .baudRate = 115200,
        .dataBits = 8,
        .stopBits = 1,
        .parity = 0,
    };
    ret = IoTUartInit(HI_UART_IDX_1, &uart_attr);
    if (ret != IOT_SUCCESS) {
        printf("Init Uart1 Falied Error No : %d\n", ret);
        return;
    }
}



