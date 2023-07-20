#include "LED.h"

//extern float lux;

void LED_Init(void)
{
    //unsigned int i;

    //初始化GPIO
    IoTGpioInit(WHITELIGHT_GPIO_PWM);
    IoTGpioInit(YELLOWLIGHT_GPIO_PWM);

    //设置GPIO_2引脚复用功能为PWM
    IoTGpioSetFunc(WHITELIGHT_GPIO_PWM, IOT_GPIO_FUNC_GPIO_7_PWM0_OUT);
    IoTGpioSetFunc(YELLOWLIGHT_GPIO_PWM, IOT_GPIO_FUNC_GPIO_10_PWM1_OUT);
    
    //设置GPIO_2引脚为输出模式
    IoTGpioSetDir(WHITELIGHT_GPIO_PWM, IOT_GPIO_DIR_OUT);
    IoTGpioSetDir(YELLOWLIGHT_GPIO_PWM, IOT_GPIO_DIR_OUT);

    //初始化PWM2端口
    IoTPwmInit(0);
    IoTPwmInit(1);
    IoTPwmStart(1, 99, 80000);
    IoTPwmStart(0, 99, 80000);
    //while (1)
    //{   
        /*
        for (i = 0; i < 100; i += 1)
        {
            //输出不同占空比的PWM波
            IoTPwmStart(1, i, 5000);
            IoTPwmStart(3, 99-i, 5000);
            usleep(10000);
        }
       
        for (i = 99; i > 0; i -= 1)
        {
            //输出不同占空比的PWM波
            IoTPwmStart(1, i, 5000);
            IoTPwmStart(3, 99-i, 5000);
            usleep(10000);
        }*/
       /* if (lux >= 1000) 
        {
            IoTPwmStart(1, 100, 80000);
            IoTPwmStart(3, 0, 80000);
        }
        else
        {
            IoTPwmStart(1, lux/10, 80000);
            IoTPwmStart(3, 99-(lux/10), 80000);
        }*/
    //}
    
}

void LED_Changelighting(unsigned int input, float dutybegin,
  float dutyend, float speed)
{
    dutybegin = (100 - dutybegin);
    dutyend = (100 - dutyend);

    if (speed == 0)
    {
        printf("lllllllllllllllllllllllllllllllllll");
        if (input == YELLOW) IoTPwmStart(1, dutyend, 80000);
        if (input == WHITE) IoTPwmStart(0, dutyend, 80000);
        if (input == ALL)
        {
            printf("lkkokokkokokokookokokok");
            IoTPwmStart(0, dutyend, 80000);
            IoTPwmStart(1, dutyend, 80000);
        }
        return;
    }

    float f = (dutyend - dutybegin) / 100;
    printf("f = %f\r\n", f);
    switch (input)
    {
        case YELLOW:
            IoTPwmStart(1, dutybegin, 80000);
            
            for (int i = 0; i < 100 ; i++)
            {
                dutybegin += f;
                IoTPwmStart(1, dutybegin, 80000);
                usleep(1000000/speed);
            }
            break;

        case WHITE:
            IoTPwmStart(0, dutybegin, 80000);

            for (int i = 0; i < 100 ; i++)
            {
                dutybegin += f;
                IoTPwmStart(0, dutybegin, 80000);
                usleep(1000000/speed);
            }
            break;

        case ALL:
            IoTPwmStart(0, dutybegin, 80000);
            IoTPwmStart(1, dutybegin, 80000);

            for (int i = 0; i < 100 ; i++)
            {
                dutybegin += f;
                IoTPwmStart(0, dutybegin, 80000);
                IoTPwmStart(1, dutybegin, 80000);
                usleep(1000000/speed);
            }
            break;

    }
}




