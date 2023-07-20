#include "Motor_control.h"

//控制舵机1的运�?
void Motor_Control_Init(void)
{
    unsigned int i;
    printf(":1:\r\n");
    //初始化GPIO
    IoTGpioInit(12);
    IoTGpioInit(11);
    printf(":2:\r\n");
    //设置GPIO_12引脚复用功能为PWM
    IoTGpioSetFunc(12, IOT_GPIO_FUNC_GPIO_12_PWM3_OUT);
    IoTGpioSetFunc(11, IOT_GPIO_FUNC_GPIO_11_GPIO);
    printf(":3:\r\n");
    //设置GPIO_12引脚为输出模�?
    IoTGpioSetDir(12, IOT_GPIO_DIR_OUT);
    IoTGpioSetDir(11, IOT_GPIO_DIR_OUT);
    printf(":4:\r\n");
    //初始化PWM3端口
    IoTPwmInit(3);
    
    printf(":5:\r\n");
    // IoTGpioInit(1);
    // IoTGpioInit(2);

    // IoTGpioSetFunc(1, IOT_GPIO_FUNC_GPIO_1_PWM4_OUT);
    // IoTGpioSetFunc(2, IOT_GPIO_FUNC_GPIO_2_GPIO);

    // IoTGpioSetDir(1, IOT_GPIO_DIR_OUT);

    // IoTPwmInit(3);

    // IoTGpioSetPull(2, IOT_GPIO_PULL_MAX);
    // IoTPwmStart(4, 50, 80000);
}

void MotorControl(uint8_t dir, float speed, float time, uint8_t ifstop)
{
        if (dir == RIGHT) 
        {
            IoTGpioSetOutputVal(11, IOT_GPIO_VALUE0);
            IoTPwmStart(3, speed, 100000);
        }
        else
        {
            IoTGpioSetOutputVal(11, IOT_GPIO_VALUE1);
            speed = 100 - speed;
            IoTPwmStart(3, speed, 100000);
        }

        usleep(time);
        if(ifstop) MotorStop();
    //IoTPwmStart(3, 50, 32000000);

    // int f = 80000;
    // for (int i = 0; i < 100 ; i++)
    // {
    //     IoTPwmStart(3, i, f);
    //     printf("%d, %d\n",i , f);
    //     usleep(500000);
    // }
}

void MotorStop()
{
    IoTGpioSetOutputVal(11, IOT_GPIO_VALUE0);
    IoTPwmStop(3);
    IoTGpioSetOutputVal(12, IOT_GPIO_VALUE0);
}

// static void PWMExampleEntry(void)
// {
//     osThreadAttr_t attr;

//     attr.name = "PWMTask";
//     attr.attr_bits = 0U;
//     attr.cb_mem = NULL;
//     attr.cb_size = 0U;
//     attr.stack_mem = NULL;
//     attr.stack_size = 2048;
//     attr.priority = 25;

//     if (osThreadNew((osThreadFunc_t)PWMTask, NULL, &attr) == NULL)
//     {
//         printf("Falied to create PWMTask!\n");
//     }
// }

// APP_FEATURE_INIT(PWMExampleEntry);
