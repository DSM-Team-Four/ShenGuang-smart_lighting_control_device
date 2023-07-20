/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "sample_nnie_main.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"

/* function : NNIE Forward */
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx, SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx,
    HI_BOOL bInstant);

/* function : NNIE ForwardWithBbox */
static HI_S32 SAMPLE_SVP_NNIE_ForwardWithBbox(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx, SVP_SRC_BLOB_S astBbox[],
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant);

/* function : Fill Src Data */
static HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx);

/* function : print report result */
static HI_S32 SAMPLE_SVP_NNIE_PrintReportResult(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam);

/* function : Cnn software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Cnn_SoftwareDeinit(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara);

/* function : Cnn Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Cnn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Cnn software para init */
static HI_S32 SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara);

/* function : Cnn init */
static HI_S32 SAMPLE_SVP_NNIE_Cnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg, SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara);

/* function : Cnn process */
static HI_S32 SAMPLE_SVP_NNIE_Cnn_PrintResult(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN);
static hi_void SAMPLE_SVP_NNIE_Cnn_Stop(void);

/* function : show Cnn sample(image 28x28 U8_C1) */
void SAMPLE_SVP_NNIE_Cnn(void);

hi_void SAMPLE_SVP_NNIE_Cnn_HandleSig(hi_void);
static void SAMPLE_SVP_NNIE_Segnet_Stop(void);

/* function : show Segnet sample(image 224x224 U8_C3) */
void SAMPLE_SVP_NNIE_Segnet(void);

/* function : Segnet sample signal handle */
hi_void SAMPLE_SVP_NNIE_Segnet_HandleSig(hi_void);

/* function : print detection result */
static HI_S32 SAMPLE_SVP_NNIE_Detection_PrintResult(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi,
    SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh);

/* function : FasterRcnn software deinit */
static HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_SoftwareDeinit(SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : FasterRcnn Deinit */
static HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : FasterRcnn software para init */
static HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_SoftwareInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : FasterRcnn parameter initialization */
static HI_S32 SAMPLE_SVP_NNIE_FasterRcnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstFasterRcnnCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);
static hi_void SAMPLE_SVP_NNIE_FasterRcnn_Stop(hi_void);

/* function : show fasterRcnn sample(image 1240x375 U8_C3) */
void SAMPLE_SVP_NNIE_FasterRcnn(void);

/* function :show fasterrcnn double_roipooling sample(image 224x224 U8_C3) */
void SAMPLE_SVP_NNIE_FasterRcnn_DoubleRoiPooling(void);

/* function : fasterRcnn sample signal handle */
hi_void SAMPLE_SVP_NNIE_FasterRcnn_HandleSig(hi_void);

/* function : Rfcn software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_SoftwareDeinit(SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Rfcn Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Rfcn software para init */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_SoftwareInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Rfcn init */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : roi to rect */
static HI_S32 SAMPLE_SVP_NNIE_RoiToRect(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum,
    HI_FLOAT *paf32ScoreThr, HI_BOOL bRmBg, SAMPLE_SVP_NNIE_RECT_ARRAY_S *pstRect, HI_U32 u32SrcWidth,
    HI_U32 u32SrcHeight, HI_U32 u32DstWidth, HI_U32 u32DstHeight);

/* function : Rfcn Proc */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_Proc(SAMPLE_SVP_NNIE_PARAM_S *pstParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSwParam);

/* function : Rfcn Proc */
static HI_S32 SAMPLE_SVP_NNIE_Rfcn_Proc_ViToVo(SAMPLE_SVP_NNIE_PARAM_S *pstParam,
    SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S *pstSwParam, VIDEO_FRAME_INFO_S *pstExtFrmInfo, HI_U32 u32BaseWidth,
    HI_U32 u32BaseHeight);

static HI_VOID SAMPLE_SVP_NNIE_Rfcn_Stop(hi_void);

static HI_S32 SAMPLE_SVP_NNIE_Rfcn_Pause(hi_void);

/* function : Rfcn vi to vo thread entry */
static HI_VOID *SAMPLE_SVP_NNIE_Rfcn_ViToVo(HI_VOID *pArgs);

/* function : Rfcn Vi->VO */
void SAMPLE_SVP_NNIE_Rfcn(void);

/* function : rfcn sample signal handle */
void SAMPLE_SVP_NNIE_Rfcn_HandleSig(void);

/* function : rfcn sample signal handle */
static hi_void SAMPLE_SVP_NNIE_Rfcn_Stop_File(hi_void);

void SAMPLE_SVP_NNIE_Rfcn_HandleSig_File(void);

/* function : Rfcn Read file */
void SAMPLE_SVP_NNIE_Rfcn_File(void);

/* function : SSD software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Ssd_SoftwareDeinit(SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Ssd Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Ssd_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Ssd software para init */
static HI_S32 SAMPLE_SVP_NNIE_Ssd_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Ssd init */
static HI_S32 SAMPLE_SVP_NNIE_Ssd_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S *pstSoftWareParam);

static void SAMPLE_SVP_NNIE_Ssd_Stop(void);

/* function : show SSD sample(image 300x300 U8_C3) */
void SAMPLE_SVP_NNIE_Ssd(void);

/* function : SSD sample signal handle */
void SAMPLE_SVP_NNIE_Ssd_HandleSig(void);

/* function : Yolov1 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov1 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Yolov1 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov1 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam);

static hi_void SAMPLE_SVP_NNIE_Yolov1_Stop(hi_void);

/* function : show YOLOV1 sample(image 448x448 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov1(void);

/* function : Yolov1 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov1_HandleSig(void);

/* function : Yolov2 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov2 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Yolov2 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov2 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam);

static void SAMPLE_SVP_NNIE_Yolov2_Stop(void);

/* function : show YOLOV2 sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov2(void);

/* function : Yolov2 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov2_HandleSig(void);

/* function : Yolov3 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov3 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Yolov3 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov3 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam);

static hi_void SAMPLE_SVP_NNIE_Yolov3_Stop(hi_void);

/* function : show YOLOV3 sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov3(void);

/* function : Yolov3 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov3_HandleSig(void);

/* function : Lstm Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Lstm_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParamm, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Lstm init */
static HI_S32 SAMPLE_SVP_NNIE_Lstm_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg, SAMPLE_SVP_NNIE_PARAM_S *pstLstmPara);

/* function : show Lstm sample(vector) */
void SAMPLE_SVP_NNIE_Lstm(void);

/* function : Lstm sample signal handle */
void SAMPLE_SVP_NNIE_Lstm_HandleSig(void);

/* function : Pavnet software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Pvanet_SoftwareDeinit(SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Pvanet Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Pvanet_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Pvanet software para init */
static HI_S32 SAMPLE_SVP_NNIE_Pvanet_SoftwareInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Pvanet parameter initialization */
static HI_S32 SAMPLE_SVP_NNIE_Pvanet_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstFasterRcnnCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : show Pvanet fasterRcnn sample(image 224x224 U8_C3) */
void SAMPLE_SVP_NNIE_Pvanet(void);

void SAMPLE_SVP_NNIE_Pvanet_HandleSig(void);


/* function : Yolov3tiny software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov3tiny Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel);

/* function : Yolov3tiny software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam);

/* function : Yolov3tiny init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam);

static hi_void SAMPLE_SVP_NNIE_Yolov3tiny_Stop(hi_void);

/* function : show YOLOV3tiny sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov3tiny(void);

/* function : Yolov3tiny sample signal handle */
void SAMPLE_SVP_NNIE_Yolov3tiny_HandleSig(void);


/******************************************************************************
* function : roi to rect
******************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_RoiToRect_Yolov3tiny(SVP_BLOB_S *pstDstScore,
    SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT *paf32ScoreThr,
    HI_BOOL bRmBg,SAMPLE_SVP_NNIE_RECT_ARRAY_S *pstRect,
    HI_U32 u32SrcWidth, HI_U32 u32SrcHeight,HI_U32 u32DstWidth,HI_U32 u32DstHeight);
 
 
/******************************************************************************
* function : Yolov3tiny Procession ViToVo
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_Proc_ViToVo(SAMPLE_SVP_NNIE_PARAM_S *pstParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSwParam, VIDEO_FRAME_INFO_S* pstExtFrmInfo,
    HI_U32 u32BaseWidth,HI_U32 u32BaseHeight);
/******************************************************************************
* function : Yolov3tiny vi to vo thread entry
******************************************************************************/
static HI_VOID* SAMPLE_SVP_NNIE_Yolov3tiny_ViToVo_thread(HI_VOID* pArgs);
/******************************************************************************
* function : Yolov3tiny vi to vo real time detection
******************************************************************************/
void SAMPLE_SVP_NNIE_Yolov3tiny_Vivo(void);
 
void SAMPLE_SVP_NNIE_Yolov3tiny_Vivo_HandleSig(void);

/******************************************************************************
* function : roi to rect
******************************************************************************/
HI_S32 SAMPLE_SVP_NNIE_RoiToRect_Yolov3(SVP_BLOB_S *pstDstScore,
    SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum, HI_FLOAT *paf32ScoreThr,
    HI_BOOL bRmBg,SAMPLE_SVP_NNIE_RECT_ARRAY_S *pstRect,
    HI_U32 u32SrcWidth, HI_U32 u32SrcHeight,HI_U32 u32DstWidth,HI_U32 u32DstHeight);
 
 
/******************************************************************************
* function : Yolov3 Procession ViToVo
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_Proc_ViToVo(SAMPLE_SVP_NNIE_PARAM_S *pstParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSwParam, VIDEO_FRAME_INFO_S* pstExtFrmInfo,
    HI_U32 u32BaseWidth,HI_U32 u32BaseHeight);
/******************************************************************************
* function : Yolov3 vi to vo thread entry
******************************************************************************/
static HI_VOID* SAMPLE_SVP_NNIE_Yolov3_ViToVo_thread(HI_VOID* pArgs);
/******************************************************************************
* function : Yolov3 vi to vo real time detection
******************************************************************************/
void SAMPLE_SVP_NNIE_Yolov3_Vivo(void);
 
void SAMPLE_SVP_NNIE_Yolov3_Vivo_HandleSig(void);