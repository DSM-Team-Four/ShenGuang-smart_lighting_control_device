#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_errno.h"
#include "iot_gpio.h"
#include "iot_gpio_ex.h"
#include "iot_i2c.h"
#include "iot_i2c_ex.h"
#include "hi_i2c.h"

#define I2C_TASK_STACK_SIZE 1024 * 8
#define I2C_TASK_PRIO 25

#define OLED_I2C_BAUDRATE 400000

#define WRITE_BIT           0x00
#define READ_BIT            0x01

#define BH1750_SLAVE_ADDR   0x23 // �ӻ���ַ
#define BH1750_PWR_DOWN     0x00 // �ر�ģ��
#define BH1750_PWR_ON       0x01 // ��ģ��ȴ�����ָ��
#define BH1750_RST          0x07 // �������ݼĴ���ֵ��PowerOnģʽ����Ч
#define BH1750_CON_H        0x10 // �����߷ֱ���ģʽ��1lx��120ms
#define BH1750_CON_H2       0x11 // �����߷ֱ���ģʽ��0.5lx��120ms
#define BH1750_CON_L        0x13 // �����ͷֱ���ģʽ��4lx��16ms
#define BH1750_ONE_H        0x20 // һ�θ߷ֱ���ģʽ��1lx��120ms��������ģ��ת��PowerDownģʽ
#define BH1750_ONE_H2       0x21 // һ�θ߷ֱ���ģʽ��0.5lx��120ms��������ģ��ת��PowerDownģʽ
#define BH1750_ONE_L        0x23 // һ�εͷֱ���ģʽ��4lx��16ms��������ģ��ת��PowerDownģʽ

void LinktoESP32Init();

void SendtoESP32(uint8_t *pData, uint16_t dataLen);