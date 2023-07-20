
#include "I2CtoESP32.h"

void LinktoESP32Init()
{
    IoTGpioInit(0); 
    IoTGpioSetFunc(0, IOT_GPIO_FUNC_GPIO_0_I2C1_SDA); /* gpio13复用I2C0_SDA */
    IoTGpioInit(1); /* 初始化gpio14 */
    IoTGpioSetFunc(1, IOT_GPIO_FUNC_GPIO_1_I2C1_SCL); /* gpio14复用I2C0_SCL */
    IoTI2cInit(HI_I2C_IDX_1, 400000);
}

/**
 @brief I2C写数据函数
 @param slaveAddr -[in] 从设备地址
 @param regAddr -[in] 寄存器地址
 @param pData -[in] 写入数据
 @param dataLen -[in] 写入数据长度
 @return 错误码
*/
int ESP_WriteData(uint8_t slaveAddr, uint8_t regAddr, uint8_t *pData, uint16_t dataLen)
{
    int ret;
    IotI2cData i2c_data = {0};

    if(0 != regAddr)
    {
        i2c_data.sendBuf = &regAddr;
        i2c_data.sendLen = 1;
        ret = IoTI2cWrite(HI_I2C_IDX_1, (slaveAddr << 1) | WRITE_BIT, &regAddr, 1);
        if(ret != 0)
        {
            printf("===== Error: I2C write status1 = 0x%x! =====\r\n", ret);
            return 0;
        }
    }

    i2c_data.sendBuf = pData;
    i2c_data.sendLen = dataLen;
    ret = IoTI2cWrite(HI_I2C_IDX_1, (slaveAddr << 1) | WRITE_BIT, pData, dataLen);
    if(ret != 0)
    {
        printf("===== Error: I2C write status1 = 0x%x! =====\r\n", ret);
        return 0;
    }

    return 1;
}


/**
 @brief BH1750初始化函数
 @param 无
 @return 无
*/
void SendtoESP32(uint8_t *pData, uint16_t dataLen)
{
    printf("%s", pData);
    LinktoESP32Init();
    ESP_WriteData(0x50, 0, pData, dataLen);
    ESP_WriteData(0x50, 0, pData, dataLen);
    usleep(180000); 
}