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

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "sample_comm_nnie.h"
#include "ai_infer_process.h"
#include "sample_media_ai.h"


#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define YOLO2MODEL_FILE_HAND    "/userdata/models/hand_classify/hand_detect.wk" // darknet framework wk model
//#define YOLO3TINYMODEL_FILE_HAND    "/userdata/models/hand_classify/yolo-tiny-people-10000-RGBandHighpre.wk" // darknet framework wk model
#define YOLO3TINYMODEL_FILE_HAND    "/userdata/models/hand_classify/kokoko_inst-12-add50000.wk" // darknet framework wk model
#define PIRIOD_NUM_MAX     49 // Logs are printed when the number of targets is detected
#define DETECT_OBJ_MAX     32 // detect max obj

static uintptr_t g_handModel = 0;

static HI_S32 Yolo2FdLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = Yolo2Create(&self, YOLO2MODEL_FILE_HAND);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("Yolo2FdLoad ret:%d\n", ret);

    return ret;
}

static HI_S32 Yolo3tinyFdLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = Yolo3tinyCreate(&self, YOLO3TINYMODEL_FILE_HAND);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    SAMPLE_PRT("Yolo3tinyFdLoad ret:%d\n", ret);

    return ret;
}

HI_S32 Yolo2HandDetectInit()
{
    return Yolo2FdLoad(&g_handModel);
}

HI_S32 Yolo3tinyHandDetectInit()
{
    return Yolo3tinyFdLoad(&g_handModel);
}

static HI_S32 Yolo2FdUnload(uintptr_t model)
{
    Yolo2Destory((SAMPLE_SVP_NNIE_CFG_S*)model);
    return 0;
}

static HI_S32 Yolo3tinyFdUnload(uintptr_t model)
{
    Yolo3tinyDestory((SAMPLE_SVP_NNIE_CFG_S*)model);
    return 0;
}

HI_S32 Yolo2HandDetectExit()
{
    return Yolo2FdUnload(g_handModel);
}

HI_S32 Yolo3tinyHandDetectExit()
{
    return Yolo3tinyFdUnload(g_handModel);
}

static HI_S32 Yolo2HandDetect(uintptr_t model, IVE_IMAGE_S *srcYuv, DetectObjInfo boxs[])
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    int objNum;
    int ret = Yolo2CalImg(self, srcYuv, boxs, DETECT_OBJ_MAX, &objNum);
    if (ret < 0) {
        SAMPLE_PRT("Hand detect Yolo2CalImg FAIL, for cal FAIL, ret:%d\n", ret);
        return ret;
    }

    return objNum;
}

static HI_S32 Yolo3tinyHandDetect(uintptr_t model, IVE_IMAGE_S *srcYuv, DetectObjInfo boxs[])
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;
    int objNum;
    int ret = Yolo3tinyCalImg(self, srcYuv, boxs, DETECT_OBJ_MAX, &objNum);
    if (ret < 0) {
        SAMPLE_PRT("Hand detect Yolo3tinyCalImg FAIL, for cal FAIL, ret:%d\n", ret);
        return ret;
    }

    return objNum;
}

HI_S32 Yolo2HandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[])
{
    int ret = Yolo2HandDetect(g_handModel, srcYuv, resArr);
    return ret;
}

HI_S32 Yolo3tinyHandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[])
{
    int ret = Yolo3tinyHandDetect(g_handModel, srcYuv, resArr);
    return ret;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
