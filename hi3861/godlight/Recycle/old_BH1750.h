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

#define WRITE_BIT           0x00
#define READ_BIT            0x01

#define BH1750_SLAVE_ADDR   0x23 // 从机地址
#define BH1750_PWR_DOWN     0x00 // 关闭模块
#define BH1750_PWR_ON       0x01 // 打开模块等待测量指令
#define BH1750_RST          0x07 // 重置数据寄存器值在PowerOn模式下有�?
#define BH1750_CON_H        0x10 // 连续高分辨率模式�?1lx�?120ms
#define BH1750_CON_H2       0x11 // 连续高分辨率模式�?0.5lx�?120ms
#define BH1750_CON_L        0x13 // 连续低分辨率模式�?4lx�?16ms
#define BH1750_ONE_H        0x20 // 一次高分辨率模式，1lx�?120ms，测量后模块转到PowerDown模式
#define BH1750_ONE_H2       0x21 // 一次高分辨率模式，0.5lx�?120ms，测量后模块转到PowerDown模式
#define BH1750_ONE_L        0x23 // 一次低分辨率模式，4lx�?16ms，测量后模块转到PowerDown模式

#define BH1750_GPIO_I2C_SDA 0
#define BH1750_GPIO_I2C_SCL 1


/**
 @brief I2C驱动初始�?
 @param �?
 @return �?
*/
void I2C_Init(void);

/**
 @brief I2C写数据函�?
 @param slaveAddr -[in] 从设备地址
 @param regAddr -[in] 寄存器地址
 @param pData -[in] 写入数据
 @param dataLen -[in] 写入数据长度
 @return 错误�?
*/
int I2C_WriteData(uint8_t slaveAddr, uint8_t regAddr, uint8_t *pData, uint16_t dataLen);

/**
 @brief I2C读数据函�?
 @param slaveAddr -[in] 从设备地址
 @param regAddr -[in] 寄存器地址
 @param pData -[in] 读出数据
 @param dataLen -[in] 读出数据长度
 @return 错误�?
*/
int I2C_ReadData(uint8_t slaveAddr, uint8_t regAddr, uint8_t *pData, uint16_t dataLen);

/**
 @brief BH1750初始化函�?
 @param �?
 @return �?
*/
void BH1750_Init(void);

/**
 @brief BH1750获取光强�?
 @param �?
 @return 光强�?
*/
float BH1750_ReadLightIntensity(void);

