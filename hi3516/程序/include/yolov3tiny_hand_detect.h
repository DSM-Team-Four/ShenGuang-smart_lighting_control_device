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

#ifndef YOLOV3TINY_HAND_DETECT_H
#define YOLOV3TINY_HAND_DETECT_H

#include "hi_type.h"

#if __cplusplus
extern "C" {
#endif

HI_S32 Yolo2HandDetectInit();
HI_S32 Yolo2HandDetectExit();
HI_S32 Yolo2HandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[]);

HI_S32 Yolo3tinyHandDetectInit();
HI_S32 Yolo3tinyHandDetectExit();
HI_S32 Yolo3tinyHandDetectCal(IVE_IMAGE_S *srcYuv, DetectObjInfo resArr[]);

#ifdef __cplusplus
}
#endif
#endif
