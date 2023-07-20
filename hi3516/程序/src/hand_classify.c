/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 该文件提供了基于yolov2的手部检测以及基于resnet18的手势识别，属于两个wk串行推理�???
 * 该文件提供了手部检测和手势识别的模型加载、模型卸载、模型推理以及AI flag业务处理的API接口�???
 * 若一帧图像中出现多个手，我们通过算法将最大手作为目标手送分类网进行推理�???
 * 并将目标手标记为绿色，其他手标记为红色�?
 *
 * This file provides hand detection based on yolov2 and gesture recognition based on resnet18,
 * which belongs to two wk serial inferences. This file provides API interfaces for model loading,
 * model unloading, model reasoning, and AI flag business processing for hand detection
 * and gesture recognition. If there are multiple hands in one frame of image,
 * we use the algorithm to use the largest hand as the target hand for inference,
 * and mark the target hand as green and the other hands as red.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov2_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

// #define HAND_FRM_WIDTH     640
// #define HAND_FRM_HEIGHT    384
#define HAND_FRM_WIDTH     416
#define HAND_FRM_HEIGHT    416
//  #define HAND_FRM_WIDTH     1920
//  #define HAND_FRM_HEIGHT    1080
#define DETECT_OBJ_MAX     32
#define RET_NUM_MAX        4
#define DRAW_RETC_THICK    2    // Draw the width of the line
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32
#define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT       224
#define MODEL_FILE_GESTURE    "/userdata/models/hand_classify/hand_gesture.wk" // darknet framework wk model

#define NOTHINGDETECT      "----"
#define FIRST_OTHERS       "1111"
#define FIRST_READING      "2222"
#define FIRST_WATCHING     "3333"
#define FIRST_SLEEPING     "4444"
#define SECOND_OTHERS      "5555"
#define SECOND_READING     "6666"
#define SECOND_WATCHING    "7777"
#define SECOND_SLEEPING    "8888"
#define HAND_CLOSED        "9999"
#define HAND_OPENING       "0000"
#define HAND_MOVING_TO_NONE          "FFFF"
#define HAND_MOVING_TO_LEFT_SPEED1   "AAAA"   
#define HAND_MOVING_TO_LEFT_SPEED2   "BBBB"  
#define HAND_MOVING_TO_LEFT_SPEED3   "CCCC"  
#define HAND_MOVING_TO_LEFT_SPEED4   "DDDD"  
#define HAND_MOVING_TO_LEFT_SPEED5   "EEEE"  
#define HAND_MOVING_TO_RIGHT_SPEED1  "aaaa"  
#define HAND_MOVING_TO_RIGHT_SPEED2  "bbbb"  
#define HAND_MOVING_TO_RIGHT_SPEED3  "cccc"  
#define HAND_MOVING_TO_RIGHT_SPEED4  "dddd"  
#define HAND_MOVING_TO_RIGHT_SPEED5  "eeee"  


static int biggestBoxIndex;
static IVE_IMAGE_S img;
static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
static RectBox boxs[DETECT_OBJ_MAX] = {0};
static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;
int uartFd = 0;

/*
 * Yolo2加载手部检测和手势分类模型
 * Load hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    Yolo2HandDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load hand detect claasify model success\n");
    /*
     * Uart串口初始�???
     * Uart open init
     */
    uartFd = UartOpenInit();
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}

/*
 * Yolo2卸载手部检测和手势分类模型
 * Unload hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    Yolo2HandDetectExit(); // Uninitialize the hand detection model
    close(uartFd);
    SAMPLE_PRT("Unload hand detect claasify model success\n");

    return 0;
}

/*
 * Yolo3tiny加载手部检测和手势分类模型
 * Load hand detect and classify model
 */
HI_S32 Yolo3tinyHandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    Yolo3tinyHandDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load hand detect claasify model success\n");
    /*
     * Uart串口初始�???
     * Uart open init
     */
    uartFd = UartOpenInit();
    if (uartFd < 0) {
        printf("uart1 open failed\r\n");
    } else {
        printf("uart1 open successed\r\n");
    }
    return ret;
}

/*
 * Yolo3tiny卸载手部检测和手势分类模型
 * Unload hand detect and classify model
 */
HI_S32 Yolo3tinyHandDetectResnetClassifyUnload(uintptr_t model)
{
    //CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    Yolo3tinyHandDetectExit(); // Uninitialize the hand detection model
    close(uartFd);
    SAMPLE_PRT("Unload hand detect claasify model success\n");

    return 0;
}

/*
 * 获得最大的�???
 * Get the maximum hand
 */
static HI_S32 GetBiggestHandIndex(RectBox boxs[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
    HI_S32 biggestBoxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
        HI_S32 boxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = boxs[biggestBoxIndex].xmax - boxs[biggestBoxIndex].xmin + 1;
        biggestBoxHeight = boxs[biggestBoxIndex].ymax - boxs[biggestBoxIndex].ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}

/*
 * 手势识别信息
 * Hand gesture recognition info
 */
static void HandDetectFlag(const RecogNumInfo resBuf)
{
    HI_CHAR *gestureName = NULL;
    switch (resBuf.num) {
        case 0u:
            gestureName = "gesture fist";
            UartSendRead(uartFd, FistGesture); // 拳头手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 1u:
            gestureName = "gesture indexUp";
            UartSendRead(uartFd, ForefingerGesture); // 食指手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 2u:
            gestureName = "gesture OK";
            UartSendRead(uartFd, OkGesture); // OK手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 3u:
            gestureName = "gesture palm";
            UartSendRead(uartFd, PalmGesture); // 手掌手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 4u:
            gestureName = "gesture yes";
            UartSendRead(uartFd, YesGesture); // yes手势
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 5u:
            gestureName = "gesture pinchOpen";
            UartSendRead(uartFd, ForefingerAndThumbGesture); // 食指 + 大拇�???
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 6u:
            gestureName = "gesture phoneCall";
            UartSendRead(uartFd, LittleFingerAndThumbGesture); // 大拇�??? + 小拇�???
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        default:
            gestureName = "gesture others";
            UartSendRead(uartFd, InvalidGesture); // 无效�???
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
            
    }
    SAMPLE_PRT("hand gesture success\n");
}

/*
 * Yolo2手部检测和手势分类推理
 * Hand detect and classify calculation
 */
HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum;
    int ret;
    int num = 0;

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "hand detect for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = Yolo2HandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        SAMPLE_PRT("yolo2_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;
    }
    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    SAMPLE_PRT("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    /*
     * 当检测到对象时，在DSTFRM中绘制一个矩�???
     * When an object is detected, a rectangle is drawn in the DSTFRM
     */
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_GREEN, DRAW_RETC_THICK); // Target hand objnum is equal to 1

        for (int j = 0; (j < objNum) && (objNum > 1); j++) {
            if (j != biggestBoxIndex) {
                remainingBoxs[num++] = boxs[j];
                /*
                 * 其他手objnum等于objnum -1
                 * Others hand objnum is equal to objnum -1
                 */
                MppFrmDrawRects(dstFrm, remainingBoxs, objNum - 1, RGB888_RED, DRAW_RETC_THICK);
            }
        }

        /*
         * 裁剪出来的图像通过预处理送分类网进行推理
         * The cropped image is preprocessed and sent to the classification network for inference
         */
        ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[biggestBoxIndex]);
        SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%#x\n", ret);

        if ((imgIn.u32Width >= WIDTH_LIMIT) && (imgIn.u32Height >= HEIGHT_LIMIT)) {
            COMPRESS_MODE_E enCompressMode = srcFrm->stVFrame.enCompressMode;
            ret = OrigImgToFrm(&imgIn, &frmIn);
            frmIn.stVFrame.enCompressMode = enCompressMode;
            SAMPLE_PRT("crop u32Width = %d, img.u32Height = %d\n", imgIn.u32Width, imgIn.u32Height);
            ret = MppFrmResize(&frmIn, &frmDst, IMAGE_WIDTH, IMAGE_HEIGHT);
            ret = FrmToOrigImg(&frmDst, &imgDst);
            ret = CnnCalImg(self,  &imgDst, numInfo, sizeof(numInfo) / sizeof((numInfo)[0]), &resLen);
            SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnCalImg FAIL, ret=%#x\n", ret);
            HI_ASSERT(resLen <= sizeof(numInfo) / sizeof(numInfo[0]));
            HandDetectFlag(numInfo[0]);
            MppFrmDestroy(&frmDst);
        }
        IveImgDestroy(&imgIn);
    }

    return ret;
}

/*
 * Yolo3tiny手部检测和手势分类推理
 * Hand detect and classify calculation
 */
HI_S32 Yolo3tinyHandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    HI_S32 resLen = 0;
    int objNum = 0;
    int ret;
    int num = 0;

    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "hand detect for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = Yolo3tinyHandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    printf("%d", objNum);
    // if (objNum == 0)
    // {
    //     HI_CHAR messagetohi3861_0[10] = {'\0'};
    //     sprintf(messagetohi3861_0, NOTHINGDETECT);
    //     HisignallingMsgSend(uartFd, messagetohi3861_0, sizeof(messagetohi3861_0));
    //     printf("#######%s######\r\n", messagetohi3861_0);
    // }
    for (int i = 0; i < objNum; i++) {
        RectBox *box = &objs[i].box;
        HI_CHAR messagetohi3861[10] = {'\0'};
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        SAMPLE_PRT("yolo3tiny_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        int handcenter = 0;
        if (objs[i].cls == 10 || objs[i].cls == 11 )
        {
            handcenter  = (box->xmin + box->xmax) / 2;
            if (handcenter > 960-160 && handcenter < 960+160)          sprintf(messagetohi3861, HAND_MOVING_TO_NONE);
            else if (handcenter <= 960-160)
            {
                if (handcenter >= 960-320 && handcenter < 960-160)      sprintf(messagetohi3861, HAND_MOVING_TO_LEFT_SPEED1);
                else if (handcenter >= 960-480 && handcenter < 960-320) sprintf(messagetohi3861, HAND_MOVING_TO_LEFT_SPEED2);
                else if (handcenter >= 960-640 && handcenter < 960-480) sprintf(messagetohi3861, HAND_MOVING_TO_LEFT_SPEED3);
                else if (handcenter >= 960-800 && handcenter < 960-640) sprintf(messagetohi3861, HAND_MOVING_TO_LEFT_SPEED4);
                else sprintf(messagetohi3861, HAND_MOVING_TO_LEFT_SPEED5);
            }
            else
            {
                if (handcenter >= 960+160 && handcenter < 960+320)      sprintf(messagetohi3861, HAND_MOVING_TO_RIGHT_SPEED1);
                else if (handcenter >= 960+320 && handcenter < 960+480) sprintf(messagetohi3861, HAND_MOVING_TO_RIGHT_SPEED2);
                else if (handcenter >= 960+480 && handcenter < 960+640) sprintf(messagetohi3861, HAND_MOVING_TO_RIGHT_SPEED3);
                else if (handcenter >= 960+640 && handcenter < 960+800) sprintf(messagetohi3861, HAND_MOVING_TO_RIGHT_SPEED4);
                else sprintf(messagetohi3861, HAND_MOVING_TO_RIGHT_SPEED5);
            }
        }
        else
        {
           sprintf(messagetohi3861, "%d%d%d%d", objs[i].cls, objs[i].cls, objs[i].cls, objs[i].cls);
        }
    
        HisignallingMsgSend(uartFd, messagetohi3861, sizeof(messagetohi3861));
        printf("#######%s######\r\n", messagetohi3861);
        boxs[0] = *box;
        MppFrmDrawRects(dstFrm, boxs, 1, RGB888_GREEN, DRAW_RETC_THICK);
    }  
    
    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
