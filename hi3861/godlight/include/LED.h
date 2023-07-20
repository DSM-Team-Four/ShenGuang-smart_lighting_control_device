#include <stdio.h>
#include <unistd.h>
#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_pwm.h"
#include "iot_gpio.h"
#include "iot_gpio_ex.h"

#ifndef __LED_H__
#define __LED_H__

#define PWM_TASK_STACK_SIZE 512
#define PWM_TASK_PRIO 25

#define WHITELIGHT_GPIO_PWM 7
#define YELLOWLIGHT_GPIO_PWM 10

#define YELLOW 1
#define WHITE 2
#define ALL 3

void LED_Init(void);

void LED_Changelighting(unsigned int input, float dutybegin,
  float dutyend, float speed);

#endif /* __LED_H__ */