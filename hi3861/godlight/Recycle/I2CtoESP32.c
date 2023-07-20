
#include "I2CtoESP32.h"

void LinktoESP32Init()
{
    IoTGpioInit(0); 
    IoTGpioSetFunc(0, IOT_GPIO_FUNC_GPIO_0_I2C1_SDA); /* gpio13����I2C0_SDA */
    IoTGpioInit(1); /* ��ʼ��gpio14 */
    IoTGpioSetFunc(1, IOT_GPIO_FUNC_GPIO_1_I2C1_SCL); /* gpio14����I2C0_SCL */
    IoTI2cInit(HI_I2C_IDX_1, 400000);
}

/**
 @brief I2Cд���ݺ���
 @param slaveAddr -[in] ���豸��ַ
 @param regAddr -[in] �Ĵ�����ַ
 @param pData -[in] д������
 @param dataLen -[in] д�����ݳ���
 @return ������
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
 @brief BH1750��ʼ������
 @param ��
 @return ��
*/
void SendtoESP32(uint8_t *pData, uint16_t dataLen)
{
    printf("%s", pData);
    LinktoESP32Init();
    ESP_WriteData(0x50, 0, pData, dataLen);
    ESP_WriteData(0x50, 0, pData, dataLen);
    usleep(180000); 
}