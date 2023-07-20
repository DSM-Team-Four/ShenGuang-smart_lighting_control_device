#include <stdio.h>

#include <unistd.h>

#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_pwm.h"
#include "iot_gpio.h"
#include "iot_gpio_ex.h"

#define PWM_TASK_STACK_SIZE 512
#define PWM_TASK_PRIO 25

#define WHITELIGHT_GPIO_PWM 0

#define LEFT 1
#define RIGHT 0


void Motor_Control_lig_Init(void);

void MotorStatus(int umotor);
