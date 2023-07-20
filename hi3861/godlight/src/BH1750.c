#include "BH1750.h"

float lux = 0.0;

/**
 @brief I2C驱动初始化
 @param 无
 @return 无
*/
void I2C_Init(void)
{
    printf("WDWdwdwdwdwdddwddwdwwdwdw\r\n");
    IoTGpioInit(BH1750_GPIO_I2C_SDA);
    IoTGpioInit(BH1750_GPIO_I2C_SCL);

    //GPIO_3复用为I2C1_SDA
    IoTGpioSetFunc(BH1750_GPIO_I2C_SDA, IOT_GPIO_FUNC_GPIO_0_I2C1_SDA);

    //GPIO_1复用为I2C1_SCL
    IoTGpioSetFunc(BH1750_GPIO_I2C_SCL, IOT_GPIO_FUNC_GPIO_1_I2C1_SCL);

    //baudrate: 400kbps
    IoTI2cInit(HI_I2C_IDX_1, 400000);

    printf("WDWdwdeofkofkoefw\r\n");
}

/**
 @brief I2C写数据函数
 @param slaveAddr -[in] 从设备地址
 @param regAddr -[in] 寄存器地址
 @param pData -[in] 写入数据
 @param dataLen -[in] 写入数据长度
 @return 错误码
*/
int I2C_WriteData(uint8_t slaveAddr, uint8_t regAddr, uint8_t *pData, uint16_t dataLen)
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
 @brief I2C读数据函数
 @param slaveAddr -[in] 从设备地址
 @param regAddr -[in] 寄存器地址
 @param pData -[in] 读出数据
 @param dataLen -[in] 读出数据长度
 @return 错误码
*/
int I2C_ReadData(uint8_t slaveAddr, uint8_t regAddr, uint8_t *pData, uint16_t dataLen)
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
            printf("===== Error: I2C write status = 0x%x! =====\r\n", ret);
            return 0;
        }
    }

    i2c_data.receiveBuf = pData;
    i2c_data.receiveLen = dataLen;
    ret = IoTI2cRead(HI_I2C_IDX_1, (slaveAddr << 1) | READ_BIT, pData, dataLen);
    if(ret != 0)
    {
        printf("===== Error: I2C read status = 0x%x! =====\r\n", ret);
        return 0;
    }

    return 1;
}

/**
 @brief BH1750初始化函数
 @param 无
 @return 无
*/
void BH1750_Init(void)
{
    I2C_Init();
    uint8_t data;
    data = BH1750_PWR_ON;              // 发送启动命令
    I2C_WriteData(BH1750_SLAVE_ADDR, 0, &data, 1);
    data = BH1750_CON_H;               // 设置连续高分辨率模式，1lx，120ms
    I2C_WriteData(BH1750_SLAVE_ADDR, 0, &data, 1);
    usleep(180000); 
}

/**
 @brief BH1750获取光强度
 @param 无
 @return 光强度
*/
float BH1750_ReadLightIntensity(void)
{
    float lux = 0.0;
    uint8_t sensorData[2] = {0};
    I2C_ReadData(BH1750_SLAVE_ADDR, 0, sensorData, 2);
    lux = (sensorData[0] << 8 | sensorData[1]) / 1.2;
    return lux;
}

// static void I2CExampleEntry(void)
// {
//     osThreadAttr_t attr;
//     attr.name = "I2CTask";
//     attr.attr_bits = 0U;
//     attr.cb_mem = NULL;
//     attr.cb_size = 0U;
//     attr.stack_mem = NULL;
//     attr.stack_size = I2C_TASK_STACK_SIZE;
//     attr.priority = I2C_TASK_PRIO;

//     if (osThreadNew((osThreadFunc_t)I2CTask, NULL, &attr) == NULL)
//     {
//         printf("Falied to create I2CTask!\n");
//     }
// }

// APP_FEATURE_INIT(I2CExampleEntry);
