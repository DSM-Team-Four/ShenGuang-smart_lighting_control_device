
/*
	接口针脚使用说明�?
	GPIO3/GPIO4分别复用�? I2C1 �? SDA/SCL , 用于BH1750通信
	GPIO14/GPIO13分别复用�? I2C0 �? SDA/SCL , 用于OLED通信
	GPIO0/GPIO1分别复用�? UART1 �? TXD/RND , 用于与Taurus通信
	GPIO7 复用�? PWM0 , 控制白灯
	GPIO8 复用�? PWM1 , 控制黄灯
	GPIO12 复用�? PWM3 , �? GPIO11 一起控制电�?
*/

#include <stdio.h>
#include <unistd.h>

#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_pwm.h"

#include "lwip/sockets.h"
#include "wifi_connect.h"

#include "BH1750.h"
#include "Motor_control.h"
#include "LED.h"
#include "uart_func.h"
//#include "oled_ssd13.h"
//#include "Motor_controll.h"
//#include "I2CtoESP32.h"

#define _PROT_ 8888
#define TCP_BACKLOG 10

#define WIFISSID "Xiaomi_AX3000"
//"xz1c_phone"
#define WIFIPSK "13940716780"
//"1234567890"

int sock_fd, new_fd;
char recvbuf[1024];
char buf[10] = {'\0'};


//motor舵机�?20ms�?(20000us)高电平的时间，用于设置占空比�?
int umotor = 0;

bool IFCONNECTEDWIFI = FALSE;
bool IFBH1750INIT = FALSE;
bool IFLIGHTPWMINIT = FALSE;
bool IFMOTORINIT = FALSE;

bool SIGNAL_HAND_CLOSEDTOOPEN = FALSE;

//服务端地址信息
struct sockaddr_in server_sock;

//客户端地址信息
struct sockaddr_in client_sock;
int sin_size;

struct sockaddr_in *cli_addr;

void WifiInit()
{
	//连接Wifi
	//WifiConnect("_OurEDA_OurFi", "OurEDA2021");
	//WifiConnect("test", "12345678");
	WifiConnect(WIFISSID , WIFIPSK);


	//创建socket
	if ((sock_fd = socket(AF_INET, SOCK_STREAM, 0)) == -1)
	{
		perror("socket is error\r\n");
		exit(1);
	}

	bzero(&server_sock, sizeof(server_sock));
	server_sock.sin_family = AF_INET;
	server_sock.sin_addr.s_addr = htonl(INADDR_ANY);
	server_sock.sin_port = htons(_PROT_);

	//调用bind函数绑定socket和地址
	if (bind(sock_fd, (struct sockaddr *)&server_sock, sizeof(struct sockaddr)) == -1)
	{
		perror("bind is error\r\n");
		exit(1);
	}

	//调用listen函数监听(指定port监听)
	if (listen(sock_fd, TCP_BACKLOG) == -1)
	{
		perror("listen is error\r\n");
		exit(1);
	}

	printf("start accept\n");

	IFCONNECTEDWIFI = TRUE;
}

void BH1750Init()
{
    BH1750_Init();
	IFBH1750INIT = TRUE;
}

void LightPwmInit()
{
    LED_Init();
	IFLIGHTPWMINIT = TRUE;
}

void MotorInit()
{
    //Motor_Control_lig_Init();
	Motor_Control_Init();
	IFMOTORINIT = TRUE;
}

void Hi3861DemoSystemInit()
{
    WifiInit();
    LightPwmInit();
    BH1750Init();
    MotorInit();
}

void BH1750Working()
{
	if (IFBH1750INIT == FALSE)
	{
		printf("FALSE\r\n");
		BH1750Init();
	}
	int cnt = 0;

	while(1)
	{
		printf("FALSE\r\n");
		float lux = 0.0;
        printf("test cnt: %d\r\n", cnt++);
        lux = BH1750_ReadLightIntensity();
        printf("sensor val: %.02f [Lux]\n", lux);
		sprintf(buf, "%d", (int)lux);
		printf("buf = %s", buf);
		usleep(5000000);
	}

}

void LightPwmWorking()
{
	if (IFLIGHTPWMINIT == FALSE) LightPwmInit(); 
    LED_Changelighting(WHITE, 0, 50, 10000);
	LED_Changelighting(YELLOW, 0, 50, 10000);
	LED_Changelighting(WHITE, 50, 0, 10000);
	LED_Changelighting(YELLOW, 50, 0, 10000);
	LED_Changelighting(WHITE, 0, 100, 10000);
	LED_Changelighting(YELLOW, 0, 100, 10000);
}

void MotorWorking()
{
	if (IFMOTORINIT == FALSE) MotorInit();
}

void WifiWorking()
{
	if (IFCONNECTEDWIFI == FALSE) WifiInit(); 
    //调用accept函数从队列中
	while (1)
	{
		sin_size = sizeof(struct sockaddr_in);
		printf("sin_size: %d\n", sin_size);

		if ((new_fd = accept(sock_fd, (struct sockaddr *)&client_sock, (socklen_t *)&sin_size)) == -1)
		{
			perror("accept");
			continue;
		}

		cli_addr = malloc(sizeof(struct sockaddr));

		printf("accept addr\r\n");

		if (cli_addr != NULL)
		{
			memcpy(cli_addr, &client_sock, sizeof(struct sockaddr));
		}

		//处理目标
		ssize_t ret;

		while (1)
		{
			//printf("sensor val: %d [Lux]\n", (int)lux);////
			//sprintf(buf, "%d", (int)lux);////
			//printf("%s",buf);////
			//printf("send :%d\r\n",lux);
			//send(new_fd, (int)lux, 5, 0);
			//send(new_fd, buf, strlen(buf), 0);////
			float lastlight = 0;
			float nowlight = 0;
			recv(new_fd, recvbuf, 1, 0);
			printf("begin to while. the recv :%s\r\n", recvbuf);
			if (recvbuf[0] >= 32 && recvbuf[0] <= 122)
			{
				lastlight = nowlight;
				nowlight = (int)recvbuf[0] - 32;
				LED_Changelighting(ALL, lastlight, nowlight, 100000000);
			}
			else if (recvbuf[0] == '{') MotorControl(RIGHT, 75, 10000, FALSE);
			else if (recvbuf[0] == '|') MotorStop();
			else if (recvbuf[0] == '}') MotorControl(LEFT, 75, 10000, FALSE);
			else if (recvbuf[0] == '~') MotorStop();
			//ret = recv(new_fd, recvbuf, sizeof(recvbuf), 0);
			//sprintf(buf, "%d", lux);
			//printf("%d\n", lux);
			
			//if ((ret = send(new_fd, buf, strlen(buf) + 1, 0)) == -1)
			//{
			//	perror("send : ");
			//}
			
			//perror("lux : ");
			if (strlen(buf) > 4) send(new_fd, buf, 3, 0);
            else send(new_fd, buf, strlen(buf), 0);
			printf("succeed Receive!\r\n");
			///send(new_fd, recvbuf, 2, 0);
			///printf("Stop send message.\r\n");
			//bzero(recvbuf, sizeof(recvbuf));
			
		}

		close(new_fd);
	}
}

void Workingin1others()
{
	printf("1_others\r\n");
}
void Workingin1reading()
{
	printf("1_reading\r\n");
}
void Workingin1watching()
{
	printf("1_watching\r\n");
}
void Workingin1sleeping()
{
	printf("1_sleeping\r\n");
}
void Workingin2others()
{
	printf("2_others\r\n");
}
void Workingin2reading()
{
	printf("2_reading\r\n");
}
void Workingin2watching()
{
	printf("2_watching\r\n");
}
void Workingin2sleeping()
{
	printf("2_sleeping\r\n");
}
void Workinginhandclosed()
{
	if (SIGNAL_HAND_CLOSEDTOOPEN == TRUE)
	{
		LED_Changelighting(ALL, 100, 0, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = FALSE;
	}
}
void Workinginhandopen()
{
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}
void Workinginmovingtonone()
{
	printf("movingtonone\r\n");
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoleft1()
{
	MotorControl(RIGHT, 40, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoleft2()
{
	MotorControl(LEFT, 60, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoleft3()
{
	MotorControl(LEFT, 70, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoleft4()
{
	MotorControl(LEFT, 80, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoleft5()
{
	MotorControl(LEFT, 100, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoright1()
{
	MotorControl(RIGHT, 40, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoright2()
{
	MotorControl(RIGHT, 60, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoright3()
{
	MotorControl(RIGHT, 70, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoright4()
{
	MotorControl(RIGHT, 80, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void Workinginmovingtoright5()
{
	MotorControl(RIGHT, 100, 1000, TRUE);
	if (SIGNAL_HAND_CLOSEDTOOPEN == FALSE)
	{
		LED_Changelighting(ALL, 0, 100, 5000000);
		SIGNAL_HAND_CLOSEDTOOPEN = TRUE;
	}
}

void NoWorking()
{
	MotorStop();
}


static void UartTask(void)
{
    const char *data = "Hello OpenHarmony !!!\n";
    uint32_t count = 0;
    uint32_t len = 0;
    unsigned char uartReadBuff[UART_BUFF_SIZE] = {0};

    // 对UART1的一些初始化 Some initialization of UART1
    Uart1GpioInit();
    // 对UART1参数的一些配�??? Some configurations of UART1 parameters
    Uart1Config();

    while (1) {
		
		
        // 通过UART1 发送数�??? Send data through UART1
        //IoTUartWrite(HI_UART_IDX_1, (unsigned char*)data, strlen(data));
        // 通过UART1 接收数据 Receive data through UART1
        len = IoTUartRead(HI_UART_IDX_1, uartReadBuff, UART_BUFF_SIZE);
        if (len > 0) {
            // 把接收到的数据打印出�??? Print the received data
            printf("Uart Read Data is: [ %d ] %s \r\n", count, uartReadBuff);
			for (int i = 0; i < UART_BUFF_SIZE; i++)
			{
				char num = uartReadBuff[i];
				printf("%c\r\n",num);
				if (num == '1')
				{
					Workingin1others();
					break;
				}
				if (num == '2')
				{
					Workingin1reading();
					break;					
				}
				if (num == '3')
				{
					Workingin1watching();
					break;
				}
				if (num == '4')
				{
					Workingin1sleeping();
					break;
				}
				if (num == '5')
				{
					Workingin2others();
					break;					
				}
				if (num == '6')
				{
					Workingin2reading();
					break;
				}
				if (num == '7')
				{
					Workingin2watching();
					break;
				}
				if (num == '8')
				{
					Workingin2sleeping();
					break;					
				}
				if (num == '9')
				{
					Workinginhandclosed();
					break;
				}
				if (num == '0')
				{
					Workinginhandopen();
					break;
				}
				if (num == 'F')
				{
					Workinginmovingtonone();
					break;					
				}
				if (num == 'A')
				{
					Workinginmovingtoleft1();
					break;
				}
				if (num == 'B')
				{
					Workinginmovingtoleft2();
					break;
				}
				if (num == 'C')
				{
					Workinginmovingtoleft3();
					break;
				}
				if (num == 'D')
				{
					Workinginmovingtoleft4();
					break;
				}
				if (num == 'E')
				{
					Workinginmovingtoleft5();
					break;
				}
				if (num == 'a')
				{
					Workinginmovingtoright1();
					break;
				}
				if (num == 'b')
				{
					Workinginmovingtoright2();
					break;
				}
				if (num == 'c')
				{
					Workinginmovingtoright3();
					break;
				}
				if (num == 'd')
				{
					Workinginmovingtoright4();
					break;
				}
				if (num == 'e')
				{
					Workinginmovingtoright5();
					break;
				}
				if (num == '-')
				{
					NoWorking();
					break;
				}
			}
			// int cls = -1; int xx = 0;
			// printf("cls=%d", cls);
			// if (uartReadBuff[3] > '0' && uartReadBuff[3] < '5') LED_Changelighting(WHITE, 0, 100, 10000);
			// else LED_Changelighting(WHITE, 0, 0, 0);
			// if (uartReadBuff[3] > '4' && uartReadBuff[3] < '9') LED_Changelighting(YELLOW, 0, 100, 0); 
			// else LED_Changelighting(YELLOW, 0, 0, 0);
        }
		// else
		// {
		// 	LED_Changelighting(WHITE, 0, 0, 0);
		// 	LED_Changelighting(YELLOW, 0, 0, 0);
		// }
        count++;
    }
}

static void Hi3861DemoMain(void)
{
	osThreadAttr_t attr;

	attr.name = "WifiWorking";
	attr.attr_bits = 0U;
	attr.cb_mem = NULL;
	attr.cb_size = 0U;
	attr.stack_mem = NULL;
	attr.stack_size = 10240;
	attr.priority = osPriorityNormal2;

	if (osThreadNew((osThreadFunc_t)WifiWorking, NULL, &attr) == NULL)
	{
		printf("[WifiWorking] Falied to create WifiWorking!\n");
	}

	attr.name = "MotorWorking";
	if (osThreadNew((osThreadFunc_t)MotorWorking, NULL, &attr) == NULL)
    {
        printf("Falied to create MotorWorking!\n");
    }

	attr.name = "LightPwmWorking";

    if (osThreadNew((osThreadFunc_t)LightPwmWorking, NULL, &attr) == NULL)
    {
        printf("Falied to create LightPwmWorking!\n");
    }

	// attr.name = "BH1750Working";
    // if (osThreadNew((osThreadFunc_t)BH1750Working, NULL, &attr) == NULL)
    // {
    //     printf("Falied to create BH1750Working!\n");
    // }

	attr.name = "UartTask";
	if (osThreadNew((osThreadFunc_t)UartTask, NULL, &attr) == NULL) {
    	printf("[UartTask] Failed to create UartTask!\n");
    }

    // attr.name = "OledmentTask";
	// attr.priority = osPriorityNormal;
    // if (osThreadNew(OledmentTask, NULL, &attr) == NULL) {
    //     printf("[OledDemo] Falied to create OledmentTask!\n");
    // }
}

APP_FEATURE_INIT(Hi3861DemoMain);
