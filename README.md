# 神光-AI智能灯控装置

 13356 大傻猫四队	丁佩一  孙韵佳  林洋

## 第一部分  设计概述

### 设计目的

虽然越来越多的家庭将灯光和电器设备连接到网络中，但是调查表明，当前的智能语音控制灯光系统并没有真正使家居生活更加方便，因为经常会出现语音不灵的情况，让许多人选择“自己起身开灯”。况且，语音所包含的信息有延迟性。同时我们也注意到，很多人在抱怨家里出现了越来越多的遥控器，它们都控制着各自的家居，灯光，遥控器一旦丢失便控制不了。为此，我们利用人工智能模型设计了通过手势控制的台灯，并且可以通过手势控制台灯的朝向，希望为未来家居形态做出一定的探索，让设备“越来越懂人”。

### 应用领域

1. 智能家居：最直接的应用当然是家庭智能家居，手势控制＋APP操控，用户可以轻松调节灯光亮度色温，而无需寻找开关或拿起遥控器，在这里，无需额外媒介，人即终端。

2. 节能环保：手势控制的台灯还有一定的节能环保优势，用户可以更直观地调整灯光亮度，设备也可以在检测无人时自动关闭，避免过度能耗。

3. 医疗康健：对于行动不便的人群，如孕妇，患者，以及与现代生活脱钩的老人，手势控制的灯控装置能方便他们控制灯光。

4. 公共场所照明：一般来说公共场所的灯光时是固定不变的，有了手势控制后可以为大众提供更加智能化的照明。

### 主要技术特点

1. AI人体感应——通过AI摄像头检测不同人物（数据集中可分辨出两人）以向主控发送不同信号，根据程序分别调节光的色温亮度。

2. AI人体状态感知——通过AI摄像头检测不同人物当前所处状态（数据集中有 看书；看电脑；睡觉；其他 四种状态），根据程序分别调节光的色温亮度。

3. AI手势感应——通过AI摄像头检测不同的手势以及其所处位置，自动控制灯的暗灭以及自转方向和速度。

4. 传感器环境检测——利用传感器感应周围环境光照强度，自动调节台灯亮度。

5. WIFI通信与APP控制——Hi3861可以连接到家庭WLAN中，自动配网，获取IP地址。在自建的手机APP上可以输入IP地址和端口来与Hi3861建立TCP连接，在手机上控制灯光亮度和马达运动。

### 关键性能指标

为了检测速度，我们使用了Yolov3-tiny检测网并成功移植部署到Taurus开发套件上，做到了保证精准感应的同时，检测速度可达惊人的40fps，完全达到满帧运行，实时检测的目的。

AI识别精度约77%，识别错误率约5%，可达到日常使用的目的，在测试中可以达到一次识别成功，而无需多次尝试。

装置利用传感器检测周围环境的光照强度，实现自动调节台灯的亮度。无论是白天还是夜晚，无论是明亮还是昏暗的环境，神光-AI智能灯控装置都能根据环境的变化进行灵活的亮度调节，提供最适合的照明效果。

### 主要创新点

1. 成功移植，部署了Yolov3-tiny轻量检测网模型，检测速度极快，几乎满帧运行，保证灯控反应的实时性；

2. 自行采集标注了1919张两个人物不同状态的姿态和三种手势共计11 classes，在自行搭建的服务器上训练获取模型数据；

3. 充分利用检测网可以检测到物体的坐标位置的特性，自行搭建算法利用手势与摄像头的相对位置控制电动马达的方向与速度，来利用手势控制灯的转向；

4. 装置成功联网并可以通过自行编写的APP控制，实时获取周遭光照强度以及灯的状态，并且控制灯的亮度和转向，形成完整的生态闭环。

## 第二部分 系统组成及功能说明

### 整体介绍

*Tips：下一页有详细的图解*

#### 硬件层面：

包括一盏可控亮度色温的LED灯，

一个电动马达（用于操控灯的方向）

Hi3861主控

BH1750光照传感器

L1910马达驱动板

备用的Esp32主板（未作主控，仅作为接口不够用时的拓展版）

以及通过UART串口通信的AI摄像头。

与Hi3861的连接方式见图

#### 软件层面（Taurus）：

内嵌运行一套AI实时检测网识别，通过Yolov3-tiny检测网实时检测周遭环境，标记检测物体种类和所处位置并把结果发送给Pegasus系统，进行下一步处理。

#### 软件层面（Pegasus）：

Hi3861主控通过I2C通信获取BH1750的光感数据，通过计算和PWM控制LED灯带；

Hi3861主控通过PWM信号与操控GPIO11的电平控制马达驱动板L1910，以间接控制差速马达的运动；

Hi3861通过UART串口接收Taurus传来的物体检测信息，进而通过PWM0、1、3控制LED灯带和差速马达的运动，达到手势控制和人物检测反应；

Hi3861通过TCP-client连接，其本身作为server服务器连接WLAN路由器，获取到IP地址，任何连接到该局域网下的设备都可连接到Hi3861，进而用户可通过手机APP控制LED灯和马达的运动。

![img](https://alivender-assets.oss-cn-beijing.aliyuncs.com/wp-content/uploads/img/PicGoPicGoclip_image002.gif)

### 各模块介绍

|     模块名称      | 介绍                                                         |
| :---------------: | :----------------------------------------------------------- |
|    Hi3861主控     | Hi3861主控模块是整个系统的核心控制单元。                     |
|     LED灯模块     | LED灯模块由白灯带和黄灯带组成，外接需一根5V和两根PWM，通过调整两根线的PWM占空比能调整灯的亮度和色温。 |
|   差速马达模块    | 差速马达模块由一个差速马达和L1910驱动板组成，L1910可以理论输入一组VCC/GND和两组PWM/GPIO，输出两组out1/out2分别控制两个不同的马达，这里只用了一个。Hi3861可通过改变PWM占空比调节马达转速，改变GPIO的VALUE来改变马达的转向。 |
| Taurus摄像头模块  | Taurus摄像头模块搭载了自行采集数据标注训练移植的Yolov3-tiny模型，模型文件大小仅8mb，经过程序优化后识别速度极快。摄像头采集数据后通过数据转换为416x416的图像，有助于提高识别速率。Taurus模块通过UART1与Hi3861通信，实时反馈给主控人物位置，手势类型和方位。 |
|  BH1750光感模块   | BH1750光感模块通过I2C协议与Hi3861通信，经过数据接收，数据计算还原后便可得到当前光照强度数值，Hi3861可通过这些数值调节LED的明暗，并把数据发送到手机APP上。 |
| WIFI与手机APP模块 | Hi3861建立一个TCP-server，主动与家庭WLAN路由器（或手机的热点）连接，获取IP地址。任何连接到同一局域网下的手机都可以通过APP来与Hi3861建立TCP连接。我们自己编写的手机APP上有如下功能：获取当前光感数据；操控LED亮度滑条；操控马达运动的按钮；以及填写IP地址和端口的输入口。用户操作的前端在后端转换为对应字符串，发送给Hi3861，Hi3861分析字符串来判断输入数据，并给出相应反应。 |
|  ESP32 拓展模块   | 在作品制作过程中预感到接口可能会不够用，因此预留了ESP32模块作为辅助接口，ESP32可通过I2C协议与Hi3861通信，以达到拓展接口的目的。 |

## 第三部分 完成情况及性能参数

人物识别和手势方位识别在经过软件优化，图像缩小和模型优化后，检测速度在精准度稍有降低的情况下相比Yolov2检测网下训练的模型提升了233%，达到了40fps，同时wk模型文件只有8mb大小，对实时检测非常有利，有力的消除掉了灯控装置在使用时的割裂感和滞后感。

作品有完整的外壳设计和接口保留，留出接口用于连接type-c供电和SD卡。作品完成度高，从环境检测，人物互动到WIFI联网，无线控制，形成闭环生态。

## 第四部分 总结

### 可拓展之处

1. 考虑启用Taurus的红外检测功能，在黑暗环境下也可检测到人，通过动态检测来判断并自动开启灯光；

2. 鉴于对不同人物的状态，灯光需要有不同的反应，而每个人所需要的反应各不相同，因此希望在APP中加入自定义模块，随时可更改自定义检测到人物不同状态时灯的行为。

3. 在作品制作过程中，发现差速马达的旋转速度和角度多是难以控制的，因此期望在未来的制作中将马达替换成伺服电机，并由ESP32辅助控制，改进外壳结构，增强旋转的稳定性。

### 心得体会

作为三名参加了这样一个高水平赛事的小白，不同于经验丰富的学长学姐们，我们在作品制作过程中遇到了太多的坎坷和bug，不过我们还是一步一步的走到了现在。回想起这几个月，从最基础的单片机开发，一边准备期末复习一边学习Hi3861多任务系统，Linux学习，到Android APP设计，神经网络结构学习，最后的工位通宵奋战和建材打印，每个人都在拼命的学习和学以致用，终于把我们心中那个完美的作品（的30%）呈现了出来。真是一次酣畅淋漓的经历啊！这次没有时间去好好研究深度学习的检测网的工作原理，未来有时间会继续学习。永无止境！感谢帮助我们指点明路的学长们，感激不尽，感激不尽。

## 第五部分 参考文献

[HI3861学习笔记（16）——光强度GY-30(BH1750)使用](https://blog.csdn.net/qq_36347513/article/details/120692539)

[Darknet—yolov3模型训练（VOC数据集) ](https://zhuanlan.zhihu.com/p/92141879)

[Android网络功能开发(6)——TCP协议通信](https://blog.csdn.net/nanoage/article/details/127967224?ops_request_misc=&request_id=dc88268d54c345a79209da705a87ef75&biz_id=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~koosearch~default-3-127967224-null-null.142)

[基于TCP/IP协议的物联网安卓应用开发基础教程（Android Studio开发）](https://blog.csdn.net/weixin_43351158/article/details/125456968?ops_request_misc=&request_id=dc88268d54c345a79209da705a87ef75&biz_id=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~koosearch~default-5-125456968-null-null.142)

[Android TCP客户端](https://blog.csdn.net/Mr_robot_strange/article/details/127771536?ops_request_misc=&request_id=dc88268d54c345a79209da705a87ef75&biz_id=&utm_medium=distribute.pc_search_result.none-task-blog-2~all~koosearch~default-9-127771536-null-null.142)

## 第六部分 附录

1. TCP连接，UART通讯方式如何保证有效性？在开发过程中，我们发现WIFI连接和UART连接都存在一定可能的消息接发滞后、丢失和错误。误差无法避免，只能寻找减少误差的优化办法。为此我们选择了——只发一个字节。即字节之间不存在任何联系，一个字节就代表了一个完整的命令。这样即使丢失数据是命令的全体丢失，只要设备多发几次，就能保证数据正常到达，而不会让主控收到残缺的命令指示产生误解。例如，手机APP上滑块调节灯的亮度，发送的字节是ASCII码 32 到 122，逆时针，顺时针分别发送 ‘{’ , ‘|’, ‘}’, ‘~’。（四个字符是因为分别是起始标志和结束标志）

![img](https://alivender-assets.oss-cn-beijing.aliyuncs.com/wp-content/uploads/img/PicGoclip_image002.jpg "Hi3861中实现信号识别的部分代码")

同理，在Hi3861通过UART与Taurus通信时，Taurus会对物体种类和坐标进行数据处理成一个字节，再将其发送给Hi3861，而不是发送五条信息（种类和xy坐标），避免了因数据缺失导致的解析错误。

![img](https://alivender-assets.oss-cn-beijing.aliyuncs.com/wp-content/uploads/img/PicGoclip_image004.jpg)

上面的图解展示了处理字节的方式。八位字节的处理方式分以下情况：

Ⅰ当识别到的物体全部是无需标记方位的（classes = 0~10）时，该字节第一位为0，其余位直接存储物体类别信息。

Ⅱ当识别到的物体需要标记方位（classes = 11，即move移动手势）时，该字节第一位为1，其余七位从左至右表示物体所处的方位，将1920x1080的屏幕分割成七块，该位为0表示手势中心不在其对应的区域，该位为1表示手势中心在该区域。

这样，只需发送一个字节，Hi3861便可获取到全部信息，进而执行相应操作。

2. 关于安卓端APP实现TCP连接和发送接收

![img](https://alivender-assets.oss-cn-beijing.aliyuncs.com/wp-content/uploads/img/PicGoclip_image006.gif)