#include "old_BH1750.h"

float lux = 0.0;

/**
 @brief I2C椹卞姩鍒濆鍖�
 @param 鏃�
 @return 鏃�
*/
void I2C_Init(void)
{
    IoTGpioInit(BH1750_GPIO_I2C_SDA);
    IoTGpioInit(BH1750_GPIO_I2C_SCL);

    //GPIO_0澶嶇敤涓篒2C1_SDA
    IoTGpioSetFunc(BH1750_GPIO_I2C_SDA, IOT_GPIO_FUNC_GPIO_0_I2C1_SDA);

    //GPIO_1澶嶇敤涓篒2C1_SCL
    IoTGpioSetFunc(BH1750_GPIO_I2C_SCL, IOT_GPIO_FUNC_GPIO_1_I2C1_SCL);

    //baudrate: 400kbps
    IoTI2cInit(HI_I2C_IDX_1, 400000);
}

/**
 @brief I2C鍐欐暟鎹嚱鏁�
 @param slaveAddr -[in] 浠庤澶囧湴鍧€
 @param regAddr -[in] 瀵勫瓨鍣ㄥ湴鍧€
 @param pData -[in] 鍐欏叆鏁版嵁
 @param dataLen -[in] 鍐欏叆鏁版嵁闀垮害
 @return 閿欒鐮�
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
 @brief I2C璇绘暟鎹嚱鏁�
 @param slaveAddr -[in] 浠庤澶囧湴鍧€
 @param regAddr -[in] 瀵勫瓨鍣ㄥ湴鍧€
 @param pData -[in] 璇诲嚭鏁版嵁
 @param dataLen -[in] 璇诲嚭鏁版嵁闀垮害
 @return 閿欒鐮�
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
 @brief BH1750鍒濆鍖栧嚱鏁�
 @param 鏃�
 @return 鏃�
*/
void BH1750_Init(void)
{
    I2C_Init();
    uint8_t data;
    data = BH1750_PWR_ON;              // 鍙戦€佸惎鍔ㄥ懡浠�
    I2C_WriteData(BH1750_SLAVE_ADDR, 0, &data, 1);
    data = BH1750_CON_H;               // 璁剧疆杩炵画楂樺垎杈ㄧ巼妯″紡锛�1lx锛�120ms
    I2C_WriteData(BH1750_SLAVE_ADDR, 0, &data, 1);
}

/**
 @brief BH1750鑾峰彇鍏夊己搴�
 @param 鏃�
 @return 鍏夊己搴�
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
