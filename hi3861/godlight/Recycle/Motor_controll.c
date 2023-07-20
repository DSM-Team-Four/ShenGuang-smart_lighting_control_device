#include "Motor_controll.h"

//motor舵机在20ms内(20000us)高电平的时间，用于设置占空比。
//int umotor = 0;

void Motor_Control_lig_Init(void)
{
    IoTGpioInit(6);
    IoTGpioSetFunc(6, IOT_GPIO_FUNC_GPIO_6_GPIO);
    IoTGpioSetDir(6, IOT_GPIO_DIR_OUT);

    // IoTGpioInit(6);
    // IoTGpioSetFunc(6, IOT_GPIO_FUNC_GPIO_6_PWM3_OUT);
    // IoTGpioSetDir(6, IOT_GPIO_DIR_OUT);
    // IoTPwmInit(3);
}

void MotorStatus(int umotor)
{

        IoTGpioSetOutputVal(6, IOT_GPIO_VALUE1);
        usleep(umotor);
        IoTGpioSetOutputVal(6, IOT_GPIO_VALUE0);
        usleep(20000 - umotor);

        // float time = 0;
        // float dutytime = 0;
        // uint16_t duty = 0;
        // time = angle/9;
        // dutytime=(time+5)/200;
        // printf("angle:%f\n",angle);
        // printf("duty:%0.1f%%\n",dutytime*100);
        // IoTPwmStart(0,dutytime,50);
}

// void MotorStop()
// {
//     IoTGpioSetOutputVal(11, IOT_GPIO_VALUE0);
//     IoTPwmStop(3);
//     IoTGpioSetOutputVal(12, IOT_GPIO_VALUE0);
// }

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
