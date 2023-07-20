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
#include <assert.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm_svp.h"
#include "hi_comm_ive.h"
#include "sample_svp_nnie_software.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */

#define USLEEP_TIME   100 // 100: usleep time, in microseconds

#define ARRAY_SUBSCRIPT_0     0
#define ARRAY_SUBSCRIPT_1     1
#define ARRAY_SUBSCRIPT_2     2
#define ARRAY_SUBSCRIPT_3     3
#define ARRAY_SUBSCRIPT_4     4
#define ARRAY_SUBSCRIPT_5     5
#define ARRAY_SUBSCRIPT_6     6
#define ARRAY_SUBSCRIPT_7     7
#define ARRAY_SUBSCRIPT_8     8
#define ARRAY_SUBSCRIPT_9     9

#define ARRAY_SUBSCRIPT_OFFSET_1    1
#define ARRAY_SUBSCRIPT_OFFSET_2    2
#define ARRAY_SUBSCRIPT_OFFSET_3    3

#define THRESH_MIN         0.2


// sample_nnie.c文件粗暴移植(

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
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"

static hi_bool g_stop_signal = HI_FALSE;
/* cnn para */
static SAMPLE_SVP_NNIE_MODEL_S s_stCnnModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stCnnNnieParam = { 0 };
static SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S s_stCnnSoftwareParam = { 0 };
/* segment para */
static SAMPLE_SVP_NNIE_MODEL_S s_stSegnetModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stSegnetNnieParam = { 0 };
/* fasterrcnn para */
static SAMPLE_SVP_NNIE_MODEL_S s_stFasterRcnnModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stFasterRcnnNnieParam = { 0 };
static SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S s_stFasterRcnnSoftwareParam = { 0 };
static SAMPLE_SVP_NNIE_NET_TYPE_E s_enNetType;
/* rfcn para */
static SAMPLE_SVP_NNIE_MODEL_S s_stRfcnModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stRfcnNnieParam = { 0 };
static SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S s_stRfcnSoftwareParam = { 0 };
static SAMPLE_IVE_SWITCH_S s_stRfcnSwitch = { HI_FALSE, HI_FALSE };
static HI_BOOL s_bNnieStopSignal = HI_FALSE;
static pthread_t s_hNnieThread = 0;
static SAMPLE_VI_CONFIG_S s_stViConfig = { 0 };

/* ssd para */
static SAMPLE_SVP_NNIE_MODEL_S s_stSsdModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stSsdNnieParam = { 0 };
static SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S s_stSsdSoftwareParam = { 0 };
/* yolov1 para */
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov1Model = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov1NnieParam = { 0 };
static SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S s_stYolov1SoftwareParam = { 0 };
/* yolov2 para */
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov2Model = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov2NnieParam = { 0 };
static SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S s_stYolov2SoftwareParam = { 0 };
/* yolov3 para */
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov3Model = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov3NnieParam = { 0 };
static SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S s_stYolov3SoftwareParam = { 0 };
/* lstm para */
static SAMPLE_SVP_NNIE_MODEL_S s_stLstmModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stLstmNnieParam = { 0 };
/* pvanet para */
static SAMPLE_SVP_NNIE_MODEL_S s_stPvanetModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stPvanetNnieParam = { 0 };
static SAMPLE_SVP_NNIE_FASTERRCNN_SOFTWARE_PARAM_S s_stPvanetSoftwareParam = { 0 };
/* yolov3tiny para */
static SAMPLE_SVP_NNIE_MODEL_S s_stYolov3tinyModel = { 0 };
static SAMPLE_SVP_NNIE_PARAM_S s_stYolov3tinyNnieParam = { 0 };
static SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S s_stYolov3tinySoftwareParam = { 0 };
/*yolov3tiny-vedio para*/
static SAMPLE_IVE_SWITCH_S s_stYolov3tinySwitch = {HI_FALSE,HI_FALSE};
/*yolov3 para*/
static SAMPLE_IVE_SWITCH_S s_stYolov3Switch = {HI_FALSE,HI_FALSE};

/* function : NNIE Forward */
static HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx, SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx,
    HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i, j;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;

    SAMPLE_SVP_CHECK_EXPR_RET(pstProcSegIdx->u32SegIdx >= pstNnieParam->pstModel->u32NetSegNum ||
        pstInputDataIdx->u32SegIdx >= pstNnieParam->pstModel->u32NetSegNum ||
        pstNnieParam->pstModel->u32NetSegNum > SVP_NNIE_MAX_NET_SEG_NUM,
        HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstProcSegIdx->u32SegIdx(%u) and pstInputDataIdx->u32SegIdx(%u) "
        "should be less than %u, pstNnieParam->pstModel->u32NetSegNum(%u) can't be greater than %u!\n",
        pstProcSegIdx->u32SegIdx, pstInputDataIdx->u32SegIdx, pstNnieParam->pstModel->u32NetSegNum,
        pstNnieParam->pstModel->u32NetSegNum, SVP_NNIE_MAX_NET_SEG_NUM);

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType == SVP_BLOB_TYPE_SEQ_S32) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) +
                    j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    /* set input blob according to node name */
    if (pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx) {
        for (i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++) {
            for (j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++) {
                if (strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN) == 0) {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx, i);
        }
    }

    /* NNIE_Forward */
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if (bInstant) {
        /* Wait NNIE finish */
        while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(
            pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId, hSvpNnieHandle, &bFinish, HI_TRUE))) {
            usleep(100); /* sleep 100 micro_seconds */
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO, "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    u32TotalStepNum = 0;
    for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) +
                    j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

/* function : NNIE ForwardWithBbox */
static HI_S32 SAMPLE_SVP_NNIE_ForwardWithBbox(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx, SVP_SRC_BLOB_S astBbox[],
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;
    HI_U32 i, j;

    SAMPLE_SVP_CHECK_EXPR_RET(pstProcSegIdx->u32SegIdx >= pstNnieParam->pstModel->u32NetSegNum ||
        pstInputDataIdx->u32SegIdx >= pstNnieParam->pstModel->u32NetSegNum ||
        pstNnieParam->pstModel->u32NetSegNum > SVP_NNIE_MAX_NET_SEG_NUM,
        HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstProcSegIdx->u32SegIdx(%u) and pstInputDataIdx->u32SegIdx(%u) "
        "should be less than %u, pstNnieParam->pstModel->u32NetSegNum(%u) should be less than %u!\n",
        pstProcSegIdx->u32SegIdx, pstInputDataIdx->u32SegIdx, pstNnieParam->pstModel->u32NetSegNum,
        pstNnieParam->pstModel->u32NetSegNum, SVP_NNIE_MAX_NET_SEG_NUM);
    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
        pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
        pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    for (i = 0; i < pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) +
                    j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    /* set input blob according to node name */
    if (pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx) {
        for (i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++) {
            for (j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++) {
                if (strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN) == 0) {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx, i);
        }
    }
    /* NNIE_ForwardWithBbox */
    s32Ret = HI_MPI_SVP_NNIE_ForwardWithBbox(&hSvpNnieHandle, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        astBbox, pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_ForwardWithBbox failed!\n");

    if (bInstant) {
        /* Wait NNIE finish */
        while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT ==
            (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE))) {
            usleep(100); /* sleep 100 micro_seconds */
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO, "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }
    u32TotalStepNum = 0;

    for (i = 0; i < pstNnieParam->astForwardWithBboxCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++) {
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType) {
            for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++) {
                u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) +
                    j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        } else {
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

/* function : Fill Src Data */
static HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx)
{
    FILE *fp = NULL;
    HI_U32 i = 0, j = 0, n = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_U8 *pu8PicAddr = NULL;
    HI_U32 *pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;
    HI_ULONG ulSize;
    HI_CHAR path[PATH_MAX] = {0};

    /* open file */
    SAMPLE_SVP_CHECK_EXPR_RET(pstNnieCfg->pszPic == HI_NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstNnieCfg->pszPic is null!\n");
    SAMPLE_SVP_CHECK_EXPR_RET(pstInputDataIdx->u32SegIdx >= SVP_NNIE_MAX_NET_SEG_NUM, HI_INVALID_VALUE,
        SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, u32SegIdx should be less than %u!\n", SVP_NNIE_MAX_NET_SEG_NUM);
    SAMPLE_SVP_CHECK_EXPR_RET(pstInputDataIdx->u32NodeIdx >= SVP_NNIE_MAX_INPUT_NUM, HI_INVALID_VALUE,
        SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, u32NodeIdx should be less than %u!\n", SVP_NNIE_MAX_INPUT_NUM);

    SAMPLE_SVP_CHECK_EXPR_RET((strlen(pstNnieCfg->pszPic) > PATH_MAX) ||
        (realpath(pstNnieCfg->pszPic, path) == HI_NULL),
        HI_ERR_SVP_NNIE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, file_name is invalid!\n");
    fp = fopen(path, "rb");
    SAMPLE_SVP_CHECK_EXPR_RET(fp == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, open file failed!\n");

    /* get data size */
    if (SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType) {
        u32VarSize = sizeof(HI_U8);
    } else {
        u32VarSize = sizeof(HI_U32);
    }

    /* fill src data */
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType) {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++) {
            for (i = 0; i < *(pu32StepAddr + n); i++) {
                ulSize = fread(pu8PicAddr, u32Dim * u32VarSize, 1, fp);
                SAMPLE_SVP_CHECK_EXPR_GOTO(ulSize != 1, FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                    "Error,Read image file failed!\n");
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr + n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            u32TotalStepNum * u32Stride);
    } else {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        if (SVP_BLOB_TYPE_YVU420SP == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType) {
            for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++) {
                for (i = 0; i < u32Chn * u32Height / 2; i++) { /* Brightness: 1 height, Chroma: 1/2 height */
                    ulSize = fread(pu8PicAddr, u32Width * u32VarSize, 1, fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(ulSize != 1, FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                        "Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        } else if (SVP_BLOB_TYPE_YVU422SP == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType) {
            for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++) {
                for (i = 0; i < u32Height * 2; i++) { /* Brightness: 1 height, Chroma: 1 height */
                    ulSize = fread(pu8PicAddr, u32Width * u32VarSize, 1, fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(ulSize != 1, FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                        "Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        } else {
            for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++) {
                for (i = 0; i < u32Chn; i++) {
                    for (j = 0; j < u32Height; j++) {
                        ulSize = fread(pu8PicAddr, u32Width * u32VarSize, 1, fp);
                        SAMPLE_SVP_CHECK_EXPR_GOTO(ulSize != 1, FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                            "Error,Read image file failed!\n");
                        pu8PicAddr += u32Stride;
                    }
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num * u32Chn * u32Height * u32Stride);
    }

    (HI_VOID)fclose(fp);
    return HI_SUCCESS;
FAIL:

    (HI_VOID)fclose(fp);
    return HI_FAILURE;
}

/* function : print report result */
static HI_S32 SAMPLE_SVP_NNIE_PrintReportResult(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam)
{
    HI_U32 u32SegNum = pstNnieParam->pstModel->u32NetSegNum;
    HI_U32 i = 0, j = 0, k = 0, n = 0;
    HI_U32 u32SegIdx = 0, u32NodeIdx = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_CHAR acReportFileName[SAMPLE_SVP_NNIE_REPORT_NAME_LENGTH] = {'\0'};
    FILE *fp = NULL;
    HI_U32 *pu32StepAddr = NULL;
    HI_S32 *ps32ResultAddr = NULL;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;

    for (u32SegIdx = 0; u32SegIdx < u32SegNum; u32SegIdx++) {
        for (u32NodeIdx = 0; u32NodeIdx < pstNnieParam->pstModel->astSeg[u32SegIdx].u16DstNum; u32NodeIdx++) {
            s32Ret = snprintf_s(acReportFileName, SAMPLE_SVP_NNIE_REPORT_NAME_LENGTH,
                SAMPLE_SVP_NNIE_REPORT_NAME_LENGTH - 1, "seg%d_layer%d_output%d_inst.linear.hex", u32SegIdx,
                pstNnieParam->pstModel->astSeg[u32SegIdx].astDstNode[u32NodeIdx].u32NodeId, 0);
            SAMPLE_SVP_CHECK_EXPR_RET(s32Ret < 0, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                "Error,create file name failed!\n");

            fp = fopen(acReportFileName, "w");
            SAMPLE_SVP_CHECK_EXPR_RET(fp == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                "Error,open file failed!\n");

            if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].enType) {
                u32Dim = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stSeq.u32Dim;
                u32Stride = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Stride;
                pu32StepAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32,
                    pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
                ps32ResultAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32,
                    pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u64VirAddr);

                for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Num; n++) {
                    for (i = 0; i < *(pu32StepAddr + n); i++) {
                        for (j = 0; j < u32Dim; j++) {
                            s32Ret = fprintf(fp, "%08x\n", *(ps32ResultAddr + j));
                            SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret < 0, PRINT_FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                "Error,write report result file failed!\n");
                        }
                        ps32ResultAddr += u32Stride / sizeof(HI_U32);
                    }
                }
            } else {
                u32Height = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Height;
                u32Width = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Width;
                u32Chn = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].unShape.stWhc.u32Chn;
                u32Stride = pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Stride;
                ps32ResultAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32,
                    pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u64VirAddr);
                for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astDst[u32NodeIdx].u32Num; n++) {
                    for (i = 0; i < u32Chn; i++) {
                        for (j = 0; j < u32Height; j++) {
                            for (k = 0; k < u32Width; k++) {
                                s32Ret = fprintf(fp, "%08x\n", *(ps32ResultAddr + k));
                                SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret < 0, PRINT_FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                    "Error,write report result file failed!\n");
                            }
                            ps32ResultAddr += u32Stride / sizeof(HI_U32);
                        }
                    }
                }
            }
            (HI_VOID)fclose(fp);
        }
    }
    return HI_SUCCESS;

PRINT_FAIL:
    (HI_VOID)fclose(fp);
    return HI_FAILURE;
}

/* function : print detection result */
static HI_S32 SAMPLE_SVP_NNIE_Detection_PrintResult(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi,
    SVP_BLOB_S *pstClassRoiNum, HI_FLOAT f32PrintResultThresh)
{
    HI_U32 i = 0, j = 0;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias = 0;
    HI_U32 u32BboxBias = 0;
    HI_FLOAT f32Score = 0.0f;
    HI_S32 *ps32Score = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstScore->u64VirAddr);
    HI_S32 *ps32Roi = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstRoi->u64VirAddr);
    HI_S32 *ps32ClassRoiNum = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstClassRoiNum->u64VirAddr);
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;
    HI_S32 s32XMin = 0, s32YMin = 0, s32XMax = 0, s32YMax = 0;

    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++) {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * SAMPLE_SVP_NNIE_COORDI_NUM;
        /* if the confidence score greater than result threshold, the result will be printed */
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / SAMPLE_SVP_NNIE_QUANT_BASE >= f32PrintResultThresh &&
            ps32ClassRoiNum[i] != 0) {
            SAMPLE_SVP_TRACE_INFO("==== The %dth class box info====\n", i);
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++) {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / SAMPLE_SVP_NNIE_QUANT_BASE;
            if (f32Score < f32PrintResultThresh) {
                break;
            }
            s32XMin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM];
            s32YMin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + 1]; /* to get next element of this array */
            s32XMax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + 2]; /* to get next element of this array */
            s32YMax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + 3]; /* to get next element of this array */
            SAMPLE_SVP_TRACE_INFO("%d %d %d %d %f\n", s32XMin, s32YMin, s32XMax, s32YMax, f32Score);
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }
    return HI_SUCCESS;
}

/* function : Yolov1 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if ((pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0) &&
        (pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0)) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/* function : Yolov1 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* hardware deinit */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /* software deinit */
    if (pstSoftWareParam != NULL) {
        s32Ret = SAMPLE_SVP_NNIE_Yolov1_SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_SVP_NNIE_Yolov1_SoftwareDeinit failed!\n");
    }
    /* model deinit */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/* function : Yolov1 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 2;
    pstSoftWareParam->u32ClassNum = 20;
    pstSoftWareParam->u32GridNumHeight = 7;
    pstSoftWareParam->u32GridNumWidth = 7;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.5f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.2f * SAMPLE_SVP_NNIE_QUANT_BASE);

    /* Malloc assist buffer memory */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum =
        pstSoftWareParam->u32BboxNumEachGrid * pstSoftWareParam->u32GridNumHeight * pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov1_GetResultTmpBuf(pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_RET(u32TmpBufTotalSize == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_SVP_NNIE_Yolov1_GetResultTmpBuf failed!\n");
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV1_INIT", NULL, (HI_U64 *)&u64PhyAddr, (void **)&pu8VirAddr,
        u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    (HI_VOID)memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /* set each tmp buffer addr */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UINTPTR_T)pu8VirAddr);

    /* set result blob */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UINTPTR_T)pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride =
        SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum * SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)((HI_UINTPTR_T)pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr) + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/* function : Yolov1 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov1_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* init hardware para */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /* init software para */
    s32Ret = SAMPLE_SVP_NNIE_Yolov1_SoftwareInit(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_SoftwareInit failed!\n", s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Yolov1_Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

static hi_void SAMPLE_SVP_NNIE_Yolov1_Stop(hi_void)
{
    SAMPLE_SVP_NNIE_Yolov1_Deinit(&s_stYolov1NnieParam, &s_stYolov1SoftwareParam, &s_stYolov1Model);
    (HI_VOID)memset_s(&s_stYolov1NnieParam, sizeof(SAMPLE_SVP_NNIE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov1SoftwareParam, sizeof(SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S), 0,
        sizeof(SAMPLE_SVP_NNIE_YOLOV1_SOFTWARE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov1Model, sizeof(SAMPLE_SVP_NNIE_MODEL_S), 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
    printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
}

/* function : show YOLOV1 sample(image 448x448 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov1(void)
{
    const HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/dog_bike_car_448x448.bgr";
    const HI_CHAR *pcModelName = "./data/nnie_model/detection/inst_yolov1_cycle.wk";
    const HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    f32PrintResultThresh = 0.2f;
    stNnieCfg.pszPic = pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    g_stop_signal = HI_FALSE;

    /* Sys init */
    s32Ret = SAMPLE_COMM_SVP_CheckSysInit();
    SAMPLE_SVP_CHECK_EXPR_RET_VOID(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_CheckSysInit failed!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }

    /* Yolov1 Load model */
    SAMPLE_SVP_TRACE_INFO("Yolov1 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stYolov1Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV1_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Yolov1 parameter initialization */
    /* Yolov1 software parameters are set in SAMPLE_SVP_NNIE_Yolov1_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov1_SoftwareInit function are correct */
    SAMPLE_SVP_TRACE_INFO("Yolov1 parameter initialization!\n");
    s_stYolov1NnieParam.pstModel = &s_stYolov1Model.stModel;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Yolov1_ParamInit(&stNnieCfg, &s_stYolov1NnieParam, &s_stYolov1SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV1_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov1_ParamInit failed!\n");

    /* Fill src data */
    SAMPLE_SVP_TRACE_INFO("Yolov1 start!\n");

    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stYolov1NnieParam, &stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV1_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /* NNIE process(process the 0-th segment) */
    stProcSegIdx.u32SegIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov1NnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV1_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    /* software process */
    /* if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov1_GetResult
     function input data are correct */

    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Yolov1_GetResult(&s_stYolov1NnieParam, &s_stYolov1SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV1_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov1_GetResult failed!\n");

    /* print result, this sample has 21 classes:
     class 0:background     class 1:plane           class 2:bicycle
     class 3:bird           class 4:boat            class 5:bottle
     class 6:bus            class 7:car             class 8:cat
     class 9:chair          class10:cow             class11:diningtable
     class 12:dog           class13:horse           class14:motorbike
     class 15:person        class16:pottedplant     class17:sheep
     class 18:sofa          class19:train           class20:tvmonitor */
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov1_Stop();
        return;
    }
    SAMPLE_SVP_TRACE_INFO("Yolov1 result:\n");
    (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov1SoftwareParam.stDstScore, &s_stYolov1SoftwareParam.stDstRoi,
        &s_stYolov1SoftwareParam.stClassRoiNum, f32PrintResultThresh);

YOLOV1_FAIL_0:
    SAMPLE_SVP_NNIE_Yolov1_Deinit(&s_stYolov1NnieParam, &s_stYolov1SoftwareParam, &s_stYolov1Model);
    SAMPLE_COMM_SVP_CheckSysExit();
}

/* function : Yolov1 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov1_HandleSig(void)
{
    g_stop_signal = HI_TRUE;
}

/* function : Yolov2 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if ((pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0) &&
        (pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0)) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/* function : Yolov2 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* hardware deinit */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /* software deinit */
    if (pstSoftWareParam != NULL) {
        s32Ret = SAMPLE_SVP_NNIE_Yolov2_SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_SVP_NNIE_Yolov2_SoftwareDeinit failed!\n");
    }
    /* model deinit */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/* function : Yolov2 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    /* The values of the following parameters are related to algorithm principles.
        For details, see related algorithms. */
    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 5;
    pstSoftWareParam->u32ClassNum = 5;
    pstSoftWareParam->u32GridNumHeight = 13;
    pstSoftWareParam->u32GridNumWidth = 13;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.25f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0] = 1.08;
    pstSoftWareParam->af32Bias[1] = 1.19;
    pstSoftWareParam->af32Bias[2] = 3.42;
    pstSoftWareParam->af32Bias[3] = 4.41;
    pstSoftWareParam->af32Bias[4] = 6.63;
    pstSoftWareParam->af32Bias[5] = 11.38;
    pstSoftWareParam->af32Bias[6] = 9.42;
    pstSoftWareParam->af32Bias[7] = 5.11;
    pstSoftWareParam->af32Bias[8] = 16.62;
    pstSoftWareParam->af32Bias[9] = 10.52;

    /* Malloc assist buffer memory */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum =
        pstSoftWareParam->u32BboxNumEachGrid * pstSoftWareParam->u32GridNumHeight * pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf(pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_RET(u32TmpBufTotalSize == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf failed!\n");
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV2_INIT", NULL, (HI_U64 *)&u64PhyAddr, (void **)&pu8VirAddr,
        u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    (HI_VOID)memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /* set each tmp buffer addr */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UINTPTR_T)pu8VirAddr);

    /* set result blob */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UINTPTR_T)pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride =
        SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum * SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr) + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr) + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/* function : Yolov2 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov2_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* init hardware para */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /* init software para */
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_SoftwareInit(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov2_SoftwareInit failed!\n", s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov2_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

static void SAMPLE_SVP_NNIE_Yolov2_Stop(void)
{
    SAMPLE_SVP_NNIE_Yolov2_Deinit(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam, &s_stYolov2Model);
    (HI_VOID)memset_s(&s_stYolov2NnieParam, sizeof(SAMPLE_SVP_NNIE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov2SoftwareParam, sizeof(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S), 0,
        sizeof(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov2Model, sizeof(SAMPLE_SVP_NNIE_MODEL_S), 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
    printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
}

/* function : show YOLOV2 sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov2(void)
{
    const HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/street_cars_416x416.bgr";
    const HI_CHAR *pcModelName = "./data/nnie_model/detection/inst_yolov2_cycle.wk";
    const HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    f32PrintResultThresh = 0.2f;
    stNnieCfg.pszPic = pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    g_stop_signal = HI_FALSE;

    /* Sys init */
    s32Ret = SAMPLE_COMM_SVP_CheckSysInit();
    SAMPLE_SVP_CHECK_EXPR_RET_VOID(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_CheckSysInit failed!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }

    /* Yolov2 Load model */
    SAMPLE_SVP_TRACE_INFO("Yolov2 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stYolov2Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Yolov2 parameter initialization */
    /* Yolov2 software parameters are set in SAMPLE_SVP_NNIE_Yolov2_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov2_SoftwareInit function are correct */
    SAMPLE_SVP_TRACE_INFO("Yolov2 parameter initialization!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }
    s_stYolov2NnieParam.pstModel = &s_stYolov2Model.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_ParamInit(&stNnieCfg, &s_stYolov2NnieParam, &s_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov2_ParamInit failed!\n");

    /* Fill src data */
    SAMPLE_SVP_TRACE_INFO("Yolov2 start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stYolov2NnieParam, &stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /* NNIE process(process the 0-th segment) */
    stProcSegIdx.u32SegIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov2NnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /* Software process */
    /* if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov2_GetResult
     function input data are correct */

    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_GetResult(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov2_GetResult failed!\n");

    /* print result, this sample has 6 classes:
     class 0:background     class 1:Carclass           class 2:Vanclass
     class 3:Truckclass     class 4:Pedestrianclass    class 5:Cyclist */
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov2_Stop();
        return;
    }
    SAMPLE_SVP_TRACE_INFO("Yolov2 result:\n");
    (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov2SoftwareParam.stDstScore, &s_stYolov2SoftwareParam.stDstRoi,
        &s_stYolov2SoftwareParam.stClassRoiNum, f32PrintResultThresh);

YOLOV2_FAIL_0:
    SAMPLE_SVP_NNIE_Yolov2_Deinit(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam, &s_stYolov2Model);
    SAMPLE_COMM_SVP_CheckSysExit();
}

/* function : Yolov2 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov2_HandleSig(void)
{
    g_stop_signal = HI_TRUE;
}

/* function : Yolov3 software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if ((pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0) &&
        (pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0)) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/* function : Yolov3 Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* hardware deinit */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /* software deinit */
    if (pstSoftWareParam != NULL) {
        s32Ret = SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_SVP_NNIE_Yolov3_SoftwareDeinit failed!\n");
    }
    /* model deinit */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}



/* function : Yolov3 software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam)
{

    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    /* The values of the following parameters are related to algorithm principles.
        For details, see related algorithms. */
    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 3;
    pstSoftWareParam->u32ClassNum = 80;
    pstSoftWareParam->au32GridNumHeight[0] = 13;
    pstSoftWareParam->au32GridNumHeight[1] = 26;
    pstSoftWareParam->au32GridNumHeight[2] = 52;
    pstSoftWareParam->au32GridNumWidth[0] = 13;
    pstSoftWareParam->au32GridNumWidth[1] = 26;
    pstSoftWareParam->au32GridNumWidth[2] = 52;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.5f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0][0] = 116;
    pstSoftWareParam->af32Bias[0][1] = 90;
    pstSoftWareParam->af32Bias[0][2] = 156;
    pstSoftWareParam->af32Bias[0][3] = 198;
    pstSoftWareParam->af32Bias[0][4] = 373;
    pstSoftWareParam->af32Bias[0][5] = 326;
    pstSoftWareParam->af32Bias[1][0] = 30;
    pstSoftWareParam->af32Bias[1][1] = 61;
    pstSoftWareParam->af32Bias[1][2] = 62;
    pstSoftWareParam->af32Bias[1][3] = 45;
    pstSoftWareParam->af32Bias[1][4] = 59;
    pstSoftWareParam->af32Bias[1][5] = 119;
    pstSoftWareParam->af32Bias[2][0] = 10;
    pstSoftWareParam->af32Bias[2][1] = 13;
    pstSoftWareParam->af32Bias[2][2] = 16;
    pstSoftWareParam->af32Bias[2][3] = 30;
    pstSoftWareParam->af32Bias[2][4] = 33;
    pstSoftWareParam->af32Bias[2][5] = 23;

    /* Malloc assist buffer memory */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;

    SAMPLE_SVP_CHECK_EXPR_RET(SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM != pstNnieParam->pstModel->astSeg[0].u16DstNum,
        HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,pstNnieParam->pstModel->astSeg[0].u16DstNum(%d) should be %d!\n",
        pstNnieParam->pstModel->astSeg[0].u16DstNum, SAMPLE_SVP_NNIE_YOLOV3_REPORT_BLOB_NUM);
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_RET(u32TmpBufTotalSize == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_SVP_NNIE_Yolov3_GetResultTmpBuf failed!\n");
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) *
        SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV3_INIT", NULL, (HI_U64 *)&u64PhyAddr, (void **)&pu8VirAddr,
        u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    (HI_VOID)memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /* set each tmp buffer addr */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr);

    /* set result blob */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum *
        sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width =
        u32ClassNum * pstSoftWareParam->u32MaxRoiNum * SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride =
        SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/* function : Yolov3 init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* init hardware para */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /* init software para */
    s32Ret = SAMPLE_SVP_NNIE_Yolov3_SoftwareInit(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov3_SoftwareInit failed!\n", s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Yolov3_Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov3_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

static hi_void SAMPLE_SVP_NNIE_Yolov3_Stop(hi_void)
{
    SAMPLE_SVP_NNIE_Yolov3_Deinit(&s_stYolov3NnieParam, &s_stYolov3SoftwareParam, &s_stYolov3Model);
    (HI_VOID)memset_s(&s_stYolov3NnieParam, sizeof(SAMPLE_SVP_NNIE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov3SoftwareParam, sizeof(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S), 0,
        sizeof(SAMPLE_SVP_NNIE_YOLOV3_SOFTWARE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov3Model, sizeof(SAMPLE_SVP_NNIE_MODEL_S), 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
    printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
}

/* function : show YOLOV3 sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov3(void)
{
    const HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/dog_bike_car.jpg";
    const HI_CHAR *pcModelName = "./data/nnie_model/detection/inst_yolov3_cycle.wk";
    const HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    f32PrintResultThresh = 0.5f;
    stNnieCfg.pszPic = pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    g_stop_signal = HI_FALSE;

    /* Sys init */
    s32Ret = SAMPLE_COMM_SVP_CheckSysInit();
    SAMPLE_SVP_CHECK_EXPR_RET_VOID(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_CheckSysInit failed!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3_Stop();
        return;
    }

    /* Yolov3 Load model */
    SAMPLE_SVP_TRACE_INFO("Yolov3 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stYolov3Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Yolov3 parameter initialization */
    /* Yolov3 software parameters are set in SAMPLE_SVP_NNIE_Yolov3_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov3_SoftwareInit function are correct */
    SAMPLE_SVP_TRACE_INFO("Yolov3 parameter initialization!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3_Stop();
        return;
    }
    s_stYolov3NnieParam.pstModel = &s_stYolov3Model.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Yolov3_ParamInit(&stNnieCfg, &s_stYolov3NnieParam, &s_stYolov3SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3_ParamInit failed!\n");

    /* Fill src data */
    SAMPLE_SVP_TRACE_INFO("Yolov3 start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stYolov3NnieParam, &stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /* NNIE process(process the 0-th segment) */
    stProcSegIdx.u32SegIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov3NnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /* Software process */
    /* if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov3_GetResult
     function input data are correct */
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Yolov3_GetResult(&s_stYolov3NnieParam, &s_stYolov3SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3_GetResult failed!\n");

    /* print result, this sample has 81 classes:
     class 0:background      class 1:person       class 2:bicycle         class 3:car            class 4:motorbike class
     5:aeroplane class 6:bus             class 7:train        class 8:truck           class 9:boat           class
     10:traffic light class 11:fire hydrant   class 12:stop sign   class 13:parking meter  class 14:bench         class
     15:bird class 16:cat            class 17:dog         class 18:horse          class 19:sheep         class 20:cow
     class 21:elephant       class 22:bear        class 23:zebra          class 24:giraffe       class 25:backpack
     class 26:umbrella       class 27:handbag     class 28:tie            class 29:suitcase      class 30:frisbee
     class 31:skis           class 32:snowboard   class 33:sports ball    class 34:kite          class 35:baseball bat
     class 36:baseball glove class 37:skateboard  class 38:surfboard      class 39:tennis racket class 40bottle
     class 41:wine glass     class 42:cup         class 43:fork           class 44:knife         class 45:spoon
     class 46:bowl           class 47:banana      class 48:apple          class 49:sandwich      class 50orange
     class 51:broccoli       class 52:carrot      class 53:hot dog        class 54:pizza         class 55:donut
     class 56:cake           class 57:chair       class 58:sofa           class 59:pottedplant   class 60bed
     class 61:diningtable    class 62:toilet      class 63:vmonitor       class 64:laptop        class 65:mouse
     class 66:remote         class 67:keyboard    class 68:cell phone     class 69:microwave     class 70:oven
     class 71:toaster        class 72:sink        class 73:refrigerator   class 74:book          class 75:clock
     class 76:vase           class 77:scissors    class 78:teddy bear     class 79:hair drier    class 80:toothbrush */
    SAMPLE_SVP_TRACE_INFO("Yolov3 result:\n");
    (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov3SoftwareParam.stDstScore, &s_stYolov3SoftwareParam.stDstRoi,
        &s_stYolov3SoftwareParam.stClassRoiNum, f32PrintResultThresh);

YOLOV3_FAIL_0:
    SAMPLE_SVP_NNIE_Yolov3_Deinit(&s_stYolov3NnieParam, &s_stYolov3SoftwareParam, &s_stYolov3Model);
    SAMPLE_COMM_SVP_CheckSysExit();
}

/* function : Yolov3 sample signal handle */
void SAMPLE_SVP_NNIE_Yolov3_HandleSig(void)
{
    g_stop_signal = HI_TRUE;
}

/* function : Yolov3tiny software deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if ((pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0) &&
        (pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0)) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/* function : Yolov3tiny Deinit */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* hardware deinit */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /* software deinit */
    if (pstSoftWareParam != NULL) {
        s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareDeinit failed!\n");
    }
    /* model deinit */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/* function : Yolov3tiny software para init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam)
{

    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize = 0;
    HI_U32 u32DstScoreSize = 0;
    HI_U32 u32ClassRoiNumSize = 0;
    HI_U32 u32TmpBufTotalSize = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;

    /* The values of the following parameters are related to algorithm principles.
        For details, see related algorithms. */
    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 3;
    pstSoftWareParam->u32ClassNum = 12;
    pstSoftWareParam->au32GridNumHeight[0] = 13;
    pstSoftWareParam->au32GridNumHeight[1] = 26;
    pstSoftWareParam->au32GridNumWidth[0] = 13;
    pstSoftWareParam->au32GridNumWidth[1] = 26;
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.5f * SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;
    pstSoftWareParam->af32Bias[0][0] = 319;
    pstSoftWareParam->af32Bias[0][1] = 169;
    pstSoftWareParam->af32Bias[0][2] = 82;
    pstSoftWareParam->af32Bias[0][3] = 58;
    pstSoftWareParam->af32Bias[0][4] = 27;
    pstSoftWareParam->af32Bias[0][5] = 14;
    pstSoftWareParam->af32Bias[1][0] = 344;
    pstSoftWareParam->af32Bias[1][1] = 135;
    pstSoftWareParam->af32Bias[1][2] = 81;
    pstSoftWareParam->af32Bias[1][3] = 37;
    pstSoftWareParam->af32Bias[1][4] = 23;
    pstSoftWareParam->af32Bias[1][5] = 10;

    /* Malloc assist buffer memory */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;

    SAMPLE_SVP_CHECK_EXPR_RET(SAMPLE_SVP_NNIE_YOLOV3TINY_REPORT_BLOB_NUM != pstNnieParam->pstModel->astSeg[0].u16DstNum,
        HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,pstNnieParam->pstModel->astSeg[0].u16DstNum(%d) should be %d!\n",
        pstNnieParam->pstModel->astSeg[0].u16DstNum, SAMPLE_SVP_NNIE_YOLOV3TINY_REPORT_BLOB_NUM);
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov3tiny_GetResultTmpBuf(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_RET(u32TmpBufTotalSize == 0, HI_ERR_SVP_NNIE_ILLEGAL_PARAM, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_SVP_NNIE_Yolov3tiny_GetResultTmpBuf failed!\n");
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32) *
        SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV3TINY_INIT", NULL, (HI_U64 *)&u64PhyAddr, (void **)&pu8VirAddr,
        u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    (HI_VOID)memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void *)pu8VirAddr, u32TotalSize);

    /* set each tmp buffer addr */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr);

    /* set result blob */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum *
        sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width =
        u32ClassNum * pstSoftWareParam->u32MaxRoiNum * SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride =
        SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * pstSoftWareParam->u32MaxRoiNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum * pstSoftWareParam->u32MaxRoiNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr =
        SAMPLE_SVP_NNIE_CONVERT_PTR_TO_ADDR(HI_U64, pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/* function : Yolov3tiny init */
static HI_S32 SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstCfg, SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S *pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /* init hardware para */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /* init software para */
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit(pstNnieParam, pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit failed!\n", s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov3tiny_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

static hi_void SAMPLE_SVP_NNIE_Yolov3tiny_Stop(hi_void)
{
    SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam, &s_stYolov3tinyModel);
    (HI_VOID)memset_s(&s_stYolov3tinyNnieParam, sizeof(SAMPLE_SVP_NNIE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov3tinySoftwareParam, sizeof(SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S), 0,
        sizeof(SAMPLE_SVP_NNIE_YOLOV3TINY_SOFTWARE_PARAM_S));
    (HI_VOID)memset_s(&s_stYolov3tinyModel, sizeof(SAMPLE_SVP_NNIE_MODEL_S), 0, sizeof(SAMPLE_SVP_NNIE_MODEL_S));
    SAMPLE_COMM_SVP_CheckSysExit();
    printf("\033[0;31mprogram termination abnormally!\033[0;39m\n");
}

/* function : show YOLOV3tiny sample(image 416x416 U8_C3) */
void SAMPLE_SVP_NNIE_Yolov3tiny(void)
{
    const HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/image_064.jpg";
    const HI_CHAR *pcModelName = "./data/nnie_model/detection/yolo-tiny-people-10000.wk";
    const HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    f32PrintResultThresh = 0.5f;
    stNnieCfg.pszPic = pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    g_stop_signal = HI_FALSE;

    /* Sys init */
    s32Ret = SAMPLE_COMM_SVP_CheckSysInit();
    SAMPLE_SVP_CHECK_EXPR_RET_VOID(s32Ret != HI_SUCCESS, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_CheckSysInit failed!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3tiny_Stop();
        return;
    }

    /* Yolov3tiny Load model */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stYolov3tinyModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Yolov3tiny parameter initialization */
    /* Yolov3tiny software parameters are set in SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit function are correct */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny parameter initialization!\n");
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3tiny_Stop();
        return;
    }
    s_stYolov3tinyNnieParam.pstModel = &s_stYolov3tinyModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit(&stNnieCfg, &s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit failed!\n");

    /* Fill src data */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3tiny_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stYolov3tinyNnieParam, &stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /* NNIE process(process the 0-th segment) */
    stProcSegIdx.u32SegIdx = 0;
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3tiny_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov3tinyNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /* Software process */
    /* if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov3tiny_GetResult
     function input data are correct */
    if (g_stop_signal == HI_TRUE) {
        SAMPLE_SVP_NNIE_Yolov3tiny_Stop();
        return;
    }
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_GetResult(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3tiny_GetResult failed!\n");

    /* print result, this sample has 8 classes:
     class 0:1_others      class 1:1_reading       class 2:1_watching         class 3:1_sleeping
     class 4:2_others      class 5:2_reading       class 6:2_watching         class 7:2_sleeping  */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny result:\n");
    (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov3tinySoftwareParam.stDstScore, &s_stYolov3tinySoftwareParam.stDstRoi,
        &s_stYolov3tinySoftwareParam.stClassRoiNum, f32PrintResultThresh);

YOLOV3TINY_FAIL_0:
    SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam, &s_stYolov3tinyModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}

/* function : Yolov3tiny sample signal handle */
void SAMPLE_SVP_NNIE_Yolov3tiny_HandleSig(void)
{
    g_stop_signal = HI_TRUE;
}


/*
 * 函数：Cnn software参数初始化
 * function : Cnn software parameter init
 */
static HI_S32 SampleSvpNnieCnnSoftwareParaInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_U32 u32GetTopNMemSize;
    HI_U32 u32GetTopBufSize;
    HI_U32 u32GetTopFrameSize;
    HI_U32 u32TotalSize;
    HI_U32 u32ClassNum = pstCnnPara->pstModel->astSeg[0].astDstNode[0].unShape.stWhc.u32Width;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;
    HI_S32 s32Ret;

    /*
     * 获取内存大小
     * Get mem size
     */
    u32GetTopFrameSize = pstCnnSoftWarePara->u32TopN*sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
    u32GetTopNMemSize = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopFrameSize)*pstNnieCfg->u32MaxInputNum;
    u32GetTopBufSize = u32ClassNum*sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
    u32TotalSize = u32GetTopNMemSize + u32GetTopBufSize;

    /*
     * 在用户态分配MMZ内存
     * Malloc memory in user mode
     */
    s32Ret = SAMPLE_COMM_SVP_MallocMem("SAMPLE_CNN_INIT", NULL, (HI_U64*)&u64PhyAddr,
        (void**)&pu8VirAddr, u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);

    /*
     * 初始化GetTopN参数
     * Init GetTopN param
     */
    pstCnnSoftWarePara->stGetTopN.u32Num = pstNnieCfg->u32MaxInputNum;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Chn = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Height = 1;
    pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Width = u32GetTopFrameSize / sizeof(HI_U32);
    pstCnnSoftWarePara->stGetTopN.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopFrameSize);
    pstCnnSoftWarePara->stGetTopN.u64PhyAddr = u64PhyAddr;
    pstCnnSoftWarePara->stGetTopN.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

    /*
     * 初始化AssistBuf参数
     * Init AssistBuf
     */
    pstCnnSoftWarePara->stAssistBuf.u32Size = u32GetTopBufSize;
    pstCnnSoftWarePara->stAssistBuf.u64PhyAddr = u64PhyAddr + u32GetTopNMemSize;
    pstCnnSoftWarePara->stAssistBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr + u32GetTopNMemSize;

    return s32Ret;
}

/*
 * 函数：Cnn software参数去初始化
 * function : Cnn software deinit
 */
static HI_S32 SampleSvpNnieCnnSoftwareDeinit(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstCnnSoftWarePara == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstCnnSoftWarePara can't be NULL!\n");
    if (pstCnnSoftWarePara->stGetTopN.u64PhyAddr != 0 && pstCnnSoftWarePara->stGetTopN.u64VirAddr != 0) {
        SAMPLE_SVP_MMZ_FREE(pstCnnSoftWarePara->stGetTopN.u64PhyAddr,
            pstCnnSoftWarePara->stGetTopN.u64VirAddr);
        pstCnnSoftWarePara->stGetTopN.u64PhyAddr = 0;
        pstCnnSoftWarePara->stGetTopN.u64VirAddr = 0;
    }
    return s32Ret;
}

/*
 * 函数：Cnn去初始化
 * function : Cnn Deinit
 */
static HI_S32 SampleSvpNnieCnnDeinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S* pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*
     * SAMPLE_SVP_NNIE_PARAM_S参数去初始化
     * Hardware param deinit
     */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S参数去初始化
     * Software param deinit
     */
    if (pstSoftWareParam != NULL) {
        s32Ret = SampleSvpNnieCnnSoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SampleSvpNnieCnnSoftwareDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_MODEL_S参数去初始化
     * Model deinit
     */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/*
 * 函数：Cnn参数初始化
 * function : Cnn Param init
 */
static HI_S32 SampleSvpNnieCnnParamInit(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S* pstCnnSoftWarePara)
{
    HI_S32 s32Ret;
    /*
     * 初始化hardware参数
     * Init hardware param
     */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstNnieCfg, pstCnnPara);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /*
     * 初始化software参数
     * Init software param
     */
    if (pstCnnSoftWarePara != NULL) {
        s32Ret = SampleSvpNnieCnnSoftwareParaInit(pstNnieCfg, pstCnnPara, pstCnnSoftWarePara);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SampleSvpNnieCnnSoftwareParaInit failed!\n", s32Ret);
    }

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SampleSvpNnieCnnDeinit(pstCnnPara, pstCnnSoftWarePara, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SampleSvpNnieCnnDeinit failed!\n", s32Ret);
    return HI_FAILURE;
}

/*
 * 基于模型文件创建CNN模型
 * Create CNN model based mode file
 */
int CnnCreate(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile)
{
    SAMPLE_SVP_NNIE_CFG_S *self;
    HI_U32 u32PicNum = 1;
    HI_S32 s32Ret;

    self = (SAMPLE_SVP_NNIE_CFG_S*)malloc(sizeof(*self));
    HI_ASSERT(self);
    if (memset_s(self, sizeof(*self), 0x00, sizeof(*self)) != EOK) {
        HI_ASSERT(0);
    }
    /*
     * 设置配置参数
     * Set configuration parameter
     */
    self->pszPic = NULL;
    self->u32MaxInputNum = u32PicNum; // max input image num in each batch
    self->u32MaxRoiNum = 0;
    self->aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
    s_stCnnSoftwareParam.u32TopN = 5; // 5: value of the u32TopN

    /*
     * 加载CNN模型
     * Load cnn model
     */
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel((char*)modelFile, &s_stCnnModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*
     * CNN参数初始化
     * Cnn软件参数设置在SampleSvpNnieCnnSoftwareParaInit,
     * 如果用户更改了网络结构，请确保
     * SampleSvpNnieCnnSoftwareParaInit函数中的参数设置是正确的
     *
     * CNN parameter initialization
     * Cnn software parameters are set in SampleSvpNnieCnnSoftwareParaInit,
     * if user has changed net struct, please make sure the parameter settings in
     * SampleSvpNnieCnnSoftwareParaInit function are correct
     */
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stCnnNnieParam.pstModel = &s_stCnnModel.stModel;
    s32Ret = SampleSvpNnieCnnParamInit(self, &s_stCnnNnieParam, &s_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SampleSvpNnieCnnParamInit failed!\n");

    /*
     * 模型关键数据
     * Model important information
     */
    SAMPLE_PRT("model={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
        s_stCnnNnieParam.astSegData[0].astSrc[0].enType,
        s_stCnnNnieParam.astSegData[0].astSrc[0].u32Num,
        s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
        s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
        s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
        s_stCnnNnieParam.astSegData[0].astSrc[0].u32Stride);

    /*
     * 记录TskBuf地址信息
     * Record TskBuf address information
     */
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");
    *model = self;
    return 0;

    CNN_FAIL_0:
        SampleSvpNnieCnnDeinit(&s_stCnnNnieParam, &s_stCnnSoftwareParam, &s_stCnnModel);
        *model = NULL;
        return -1;
}

/*
 * 销毁CNN模型
 * Destroy CNN model
 */
void CnnDestroy(SAMPLE_SVP_NNIE_CFG_S *self)
{
    HI_S32 s32Ret;

    /*
     * 移除TskBuf地址信息
     * Remove TskBuf address information
     */
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

    CNN_FAIL_0:
        SampleSvpNnieCnnDeinit(&s_stCnnNnieParam, &s_stCnnSoftwareParam, &s_stCnnModel);
        free(self);
}

static HI_S32 FillNnieByImg(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, int segId, int nodeId, const IVE_IMAGE_S *img)
{
    HI_U32 i;
    HI_U32 j;
    HI_U32 n;
    HI_U32 u32Height = 0;
    HI_U32 u32Width = 0;
    HI_U32 u32Chn = 0;
    HI_U32 u32Stride = 0;
    HI_U32 u32VarSize;
    HI_U8 *pu8PicAddr = NULL;

    /*
     * 获取数据大小
     * Get data size
     */
    if (SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[segId].astSrc[nodeId].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
        u32VarSize = sizeof(HI_U8);
    } else {
        u32VarSize = sizeof(HI_U32);
    }

    /*
     * 填充源数据
     * Fill src data
     */
    if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
        HI_ASSERT(0);
    } else {
        u32Height = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[segId].astSrc[nodeId].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[segId].astSrc[nodeId].u32Stride;
        pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8,
            pstNnieParam->astSegData[segId].astSrc[nodeId].u64VirAddr);

        if (SVP_BLOB_TYPE_YVU420SP == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
            HI_ASSERT(pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num == 1);
            for (n = 0; n < pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num; n++) {
                // Y
                const uint8_t *srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[0];
                HI_ASSERT(srcData);
                for (j = 0; j < u32Height; j++) {
                    if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                        HI_ASSERT(0);
                    }
                    pu8PicAddr += u32Stride;
                    srcData += img->au32Stride[0];
                }
                // UV
                srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[1];
                HI_ASSERT(srcData);
                for (j = 0; j < u32Height / 2; j++) { // 2: 1/2Height
                    if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                        HI_ASSERT(0);
                    }
                    pu8PicAddr += u32Stride;
                    srcData += img->au32Stride[1];
                }
            }
        } else if (SVP_BLOB_TYPE_YVU422SP == pstNnieParam->astSegData[segId].astSrc[nodeId].enType) {
            HI_ASSERT(0);
        } else {
            for (n = 0; n < pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num; n++) {
                for (i = 0; i < u32Chn; i++) {
                    const uint8_t *srcData = (const uint8_t*)(uintptr_t)img->au64VirAddr[i];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height; j++) {
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK) {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[i];
                    }
                }
            }
        }

        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[segId].astSrc[nodeId].u64PhyAddr,
            SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[segId].astSrc[nodeId].u64VirAddr),
            pstNnieParam->astSegData[segId].astSrc[nodeId].u32Num*u32Chn*u32Height*u32Stride);
    }

    return HI_SUCCESS;
}

void CnnFetchRes(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN, RecogNumInfo resBuf[], int resSize, int* resLen)
{
    HI_ASSERT(pstGetTopN);
    HI_U32 i;
    HI_U32 j = 0;
    HI_U32 *pu32Tmp = NULL;
    HI_U32 u32Stride = pstGetTopN->u32Stride;
    if (memset_s(resBuf, resSize * sizeof(resBuf[0]), 0x00, resSize * sizeof(resBuf[0])) != EOK) {
        HI_ASSERT(0);
    }

    int resId = 0;
    pu32Tmp = (HI_U32*)((HI_UL)pstGetTopN->u64VirAddr + j * u32Stride);
    for (i = 0; i < u32TopN * 2 && resId < resSize; i += 2, resId++) { // 2: u32TopN*2
        resBuf[resId].num = pu32Tmp[i];
        resBuf[resId].score = pu32Tmp[i + 1];
    }
    *resLen = resId;
}

/*
 * 对一帧图像进行推理
 * Calculate a frame of image
 */
int CnnCalImg(SAMPLE_SVP_NNIE_CFG_S* self,
    const IVE_IMAGE_S *img, RecogNumInfo resBuf[], int resSize, int* resLen)
{
    HI_S32 s32Ret;
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*
     * 填充源数据
     * Fill src data
     */
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = FillNnieByImg(self, &s_stCnnNnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*
     * NNIE推理(process the 0-th segment)
     * NNIE process(process the 0-th segment)
     */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stCnnNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /*
     * CPU处理
     * 如果用户更改了网络结构，请确保SAMPLE_SVP_NNIE_Cnn_GetTopN函数的输入数据是正确的
     *
     * Software process
     * if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Cnn_GetTopN
     * function's input datas are correct
     */
    s32Ret = SAMPLE_SVP_NNIE_Cnn_GetTopN(&s_stCnnNnieParam, &s_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_CnnGetTopN failed!\n");

    /*
     * 打印结果
     * Print result
     */
    CnnFetchRes(&s_stCnnSoftwareParam.stGetTopN, s_stCnnSoftwareParam.u32TopN, resBuf, resSize, resLen);
    return 0;

    CNN_FAIL_1:
        return -1;
}

/*
 * 函数：Yolov2 software参数初始化
 * function : Yolov2 software param init
 */
static HI_S32 SampleSvpNnieYolov2SoftwareInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret;
    HI_U32 u32ClassNum = 0;
    HI_U32 u32BboxNum;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32DstRoiSize;
    HI_U32 u32DstScoreSize;
    HI_U32 u32ClassRoiNumSize;
    HI_U32 u32TmpBufTotalSize;
    HI_U64 u64PhyAddr = 0;
    HI_U8* pu8VirAddr = NULL;

    pstSoftWareParam->u32OriImHeight = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Height;
    pstSoftWareParam->u32OriImWidth = pstNnieParam->astSegData[0].astSrc[0].unShape.stWhc.u32Width;
    pstSoftWareParam->u32BboxNumEachGrid = 5; // 5: 2BboxNumEachGrid
    pstSoftWareParam->u32ClassNum = 1; // 5: class number
    pstSoftWareParam->u32GridNumHeight = 12; // 12: GridNumHeight
    pstSoftWareParam->u32GridNumWidth = 20; // 20: GridNumWidth
    pstSoftWareParam->u32NmsThresh = (HI_U32)(0.3f*SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32ConfThresh = (HI_U32)(0.25f*SAMPLE_SVP_NNIE_QUANT_BASE);
    pstSoftWareParam->u32MaxRoiNum = 10;  // 10: MaxRoiNum
    pstSoftWareParam->af32Bias[0] = 0.52; // 0.52: af32Bias[0] value
    pstSoftWareParam->af32Bias[1] = 0.61; // 0.61: af32Bias[1] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_2] = 1.05; // 1.05: af32Bias[ARRAY_SUBSCRIPT_2] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_3] = 1.12; // 1.12: af32Bias[ARRAY_SUBSCRIPT_3] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_4] = 1.85; // 1.85: af32Bias[ARRAY_SUBSCRIPT_4] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_5] = 2.05; // 2.05: af32Bias[ARRAY_SUBSCRIPT_5] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_6] = 4.63; // 4.63: af32Bias[ARRAY_SUBSCRIPT_6] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_7] = 4.49; // 4.49: af32Bias[ARRAY_SUBSCRIPT_7] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_8] = 7.15; // 7.15: af32Bias[ARRAY_SUBSCRIPT_8] value
    pstSoftWareParam->af32Bias[ARRAY_SUBSCRIPT_9] = 7.56; // 7.56: af32Bias[ARRAY_SUBSCRIPT_9] value

    /*
     * 申请辅助内存空间
     * Malloc assist buffer memory
     */
    u32ClassNum = pstSoftWareParam->u32ClassNum + 1;
    u32BboxNum = pstSoftWareParam->u32BboxNumEachGrid*pstSoftWareParam->u32GridNumHeight*
        pstSoftWareParam->u32GridNumWidth;
    u32TmpBufTotalSize = SAMPLE_SVP_NNIE_Yolov2_GetResultTmpBuf(pstSoftWareParam);
    u32DstRoiSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    u32DstScoreSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    u32ClassRoiNumSize = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * sizeof(HI_U32));
    u32TotalSize = u32TotalSize + u32DstRoiSize + u32DstScoreSize + u32ClassRoiNumSize + u32TmpBufTotalSize;
    s32Ret = SAMPLE_COMM_SVP_MallocCached("SAMPLE_YOLOV2_INIT", NULL, (HI_U64*)&u64PhyAddr,
        (void**)&pu8VirAddr, u32TotalSize);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,Malloc memory failed!\n");
    memset_s(pu8VirAddr, u32TotalSize, 0, u32TotalSize);
    SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void*)pu8VirAddr, u32TotalSize);

    /*
     * 设置每个tmp buffer地址
     * Set each tmp buffer addr
     */
    pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = u64PhyAddr;
    pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr);

    /*
     * 设置结果blob
     * Set result blob
     */
    pstSoftWareParam->stDstRoi.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstRoi.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize;
    pstSoftWareParam->stDstRoi.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize);
    pstSoftWareParam->stDstRoi.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum *
        u32BboxNum * sizeof(HI_U32) * SAMPLE_SVP_NNIE_COORDI_NUM);
    pstSoftWareParam->stDstRoi.u32Num = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstRoi.unShape.stWhc.u32Width = u32ClassNum *
        u32BboxNum*SAMPLE_SVP_NNIE_COORDI_NUM;

    pstSoftWareParam->stDstScore.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stDstScore.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize + u32DstRoiSize;
    pstSoftWareParam->stDstScore.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize + u32DstRoiSize);
    pstSoftWareParam->stDstScore.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum * u32BboxNum * sizeof(HI_U32));
    pstSoftWareParam->stDstScore.u32Num = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stDstScore.unShape.stWhc.u32Width = u32ClassNum*u32BboxNum;

    pstSoftWareParam->stClassRoiNum.enType = SVP_BLOB_TYPE_S32;
    pstSoftWareParam->stClassRoiNum.u64PhyAddr = u64PhyAddr + u32TmpBufTotalSize +
        u32DstRoiSize + u32DstScoreSize;
    pstSoftWareParam->stClassRoiNum.u64VirAddr = (HI_U64)((HI_UL)pu8VirAddr + u32TmpBufTotalSize +
        u32DstRoiSize + u32DstScoreSize);
    pstSoftWareParam->stClassRoiNum.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32ClassNum*sizeof(HI_U32));
    pstSoftWareParam->stClassRoiNum.u32Num = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Chn = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Height = 1;
    pstSoftWareParam->stClassRoiNum.unShape.stWhc.u32Width = u32ClassNum;

    return s32Ret;
}

/*
 * 函数：Yolov2 software参数去初始化
 * function : Yolov2 software param Deinit
 */
static HI_S32 SampleSvpNnieYolov2SoftwareDeinit(SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_CHECK_EXPR_RET(pstSoftWareParam == NULL, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, pstSoftWareParam can't be NULL!\n");
    if (pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr != 0 && pstSoftWareParam->stGetResultTmpBuf.u64VirAddr != 0) {
        SAMPLE_SVP_MMZ_FREE(pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr,
            pstSoftWareParam->stGetResultTmpBuf.u64VirAddr);
        pstSoftWareParam->stGetResultTmpBuf.u64PhyAddr = 0;
        pstSoftWareParam->stGetResultTmpBuf.u64VirAddr = 0;
        pstSoftWareParam->stDstRoi.u64PhyAddr = 0;
        pstSoftWareParam->stDstRoi.u64VirAddr = 0;
        pstSoftWareParam->stDstScore.u64PhyAddr = 0;
        pstSoftWareParam->stDstScore.u64VirAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64PhyAddr = 0;
        pstSoftWareParam->stClassRoiNum.u64VirAddr = 0;
    }
    return s32Ret;
}

/*
 * 函数：Yolov2去初始化
 * function : Yolov2 Deinit
 */
static HI_S32 SampleSvpNnieYolov2Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*
     * SAMPLE_SVP_NNIE_PARAM_S参数去初始化
     * Hardware param deinit
     */
    if (pstNnieParam != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }
    /*
     * SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S参数去初始化
     * Software deinit
     */
    if (pstSoftWareParam != NULL) {
        s32Ret = SampleSvpNnieYolov2SoftwareDeinit(pstSoftWareParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SampleSvpNnieYolov2SoftwareDeinit failed!\n");
    }

    /*
     * SAMPLE_SVP_NNIE_MODEL_S参数去初始化
     * Model deinit
     */
    if (pstNnieModel != NULL) {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/*
 * 函数：Yolov2参数初始化
 * function : Yolov2 Param init
 */
static HI_S32 SampleSvpNnieYolov2ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_YOLOV2_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret;
    /*
     * 初始化hardware参数
     * Init hardware param
     */
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg, pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

    /*
     * 初始化software参数
     * Init software param
     */
    s32Ret = SampleSvpNnieYolov2SoftwareInit(pstCfg, pstNnieParam,
        pstSoftWareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_SoftwareInit failed!\n", s32Ret);
    return s32Ret;
INIT_FAIL_0:
    s32Ret = SampleSvpNnieYolov2Deinit(pstNnieParam, pstSoftWareParam, NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_SVP_NNIE_Yolov1_Deinit failed!\n", s32Ret);
    return HI_FAILURE;
}

/*
 * 函数：基于模型文件创建Yolov2模型
 * function : Creat Yolov2 model basad mode file
 */
int Yolo2Create(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile)
{
    SAMPLE_SVP_NNIE_CFG_S *self;
    HI_U32 u32PicNum = 1;
    HI_S32 s32Ret;

    self = (SAMPLE_SVP_NNIE_CFG_S*)malloc(sizeof(*self));
    HI_ASSERT(self);
    memset_s(self, sizeof(*self), 0x00, sizeof(*self));

    /*
     * 设置配置参数
     * Set configuration parameter
     */
    self->pszPic = NULL;
    self->u32MaxInputNum = u32PicNum; // max input image num in each batch
    self->u32MaxRoiNum = 0;
    self->aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core

    /*
     * 加载Yolov2模型
     * Load Yolov2 model
     */
    SAMPLE_SVP_TRACE_INFO("Yolov2 Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel((char*)modelFile, &s_stYolov2Model);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error, SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*
     * Yolov2参数初始化
     * Yolov2软件参数设置在SampleSvpNnieYolov2SoftwareInit,
     * 如果用户更改了网络结构，请确保
     * SampleSvpNnieYolov2SoftwareInit函数中的参数设置是正确的
     *
     * Yolov2 parameter initialization
     * Yolov2 software parameters are set in SampleSvpNnieYolov2SoftwareInit,
     * if user has changed net struct, please make sure the parameter settings in
     * SampleSvpNnieYolov2SoftwareInit function are correct
     */
    SAMPLE_SVP_TRACE_INFO("Yolov2 parameter initialization!\n");
    s_stYolov2NnieParam.pstModel = &s_stYolov2Model.stModel;
    s32Ret = SampleSvpNnieYolov2ParamInit(self, &s_stYolov2NnieParam, &s_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SampleSvpNnieYolov2ParamInit failed!\n");

    /*
     * 模型关键数据
     * Model important information
     */
    SAMPLE_PRT("model.base={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
        s_stYolov2NnieParam.astSegData[0].astSrc[0].enType,
        s_stYolov2NnieParam.astSegData[0].astSrc[0].u32Num,
        s_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
        s_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
        s_stYolov2NnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
        s_stYolov2NnieParam.astSegData[0].astSrc[0].u32Stride);
    SAMPLE_PRT("model.soft={ class=%u, ori.w=%u, ori.h=%u, bnum=%u, \
        grid.w=%u, grid.h=%u, nmsThresh=%u, confThresh=%u, u32MaxRoiNum=%u }\n",
        s_stYolov2SoftwareParam.u32ClassNum,
        s_stYolov2SoftwareParam.u32OriImWidth,
        s_stYolov2SoftwareParam.u32OriImHeight,
        s_stYolov2SoftwareParam.u32BboxNumEachGrid,
        s_stYolov2SoftwareParam.u32GridNumWidth,
        s_stYolov2SoftwareParam.u32GridNumHeight,
        s_stYolov2SoftwareParam.u32NmsThresh,
        s_stYolov2SoftwareParam.u32ConfThresh,
        s_stYolov2SoftwareParam.u32MaxRoiNum);

    *model = self;
    return 0;

    YOLOV2_FAIL_0:
        SAMPLE_PRT("Yolo2Create SampleSvpNnieYolov2Deinit\n");
        SampleSvpNnieYolov2Deinit(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam, &s_stYolov2Model);
        *model = NULL;
        return -1;
}

 int Yolo3tinyCreate(SAMPLE_SVP_NNIE_CFG_S **model, const char* modelFile)
{
    const HI_U32 u32PicNum = 1;
    HI_FLOAT f32PrintResultThresh = 0.0f;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S stNnieCfg = { 0 };
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = { 0 };
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = { 0 };

    /* Set configuration parameter */
    f32PrintResultThresh = 0.5f;
    stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core

    /* Yolov3tiny Load model */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel((char*)modelFile, &s_stYolov3tinyModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /* Yolov3tiny parameter initialization */
    /* Yolov3tiny software parameters are set in SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Yolov3tiny_SoftwareInit function are correct */
    SAMPLE_SVP_TRACE_INFO("Yolov3tiny parameter initialization!\n");

    s_stYolov3tinyNnieParam.pstModel = &s_stYolov3tinyModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit(&stNnieCfg, &s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3tiny_ParamInit failed!\n");
    /*
     * 模型关键数据
     * Model important information
     */
    SAMPLE_PRT("model.base={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].enType,
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].u32Num,
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
        s_stYolov3tinyNnieParam.astSegData[0].astSrc[0].u32Stride);
    SAMPLE_PRT("model.soft={ class=%u, ori.w=%u, ori.h=%u, bnum=%u, \
        grid.w.1=%u, grid.h.1=%u, grid.w.2=%u, grid.h.2=%u, nmsThresh=%u, confThresh=%u, u32MaxRoiNum=%u }\n",
        s_stYolov3tinySoftwareParam.u32ClassNum,
        s_stYolov3tinySoftwareParam.u32OriImWidth,
        s_stYolov3tinySoftwareParam.u32OriImHeight,
        s_stYolov3tinySoftwareParam.u32BboxNumEachGrid,
        s_stYolov3tinySoftwareParam.au32GridNumWidth[0],
        s_stYolov3tinySoftwareParam.au32GridNumHeight[0],
        s_stYolov3tinySoftwareParam.au32GridNumWidth[1],
        s_stYolov3tinySoftwareParam.au32GridNumHeight[1],
        s_stYolov3tinySoftwareParam.u32NmsThresh,
        s_stYolov3tinySoftwareParam.u32ConfThresh,
        s_stYolov3tinySoftwareParam.u32MaxRoiNum);

    *model = &stNnieCfg;
    return 0;

    YOLOV3TINY_FAIL_0:
        SAMPLE_PRT("Yolo3tinyCreate SampleSvpNnieYolov3tinyDeinit\n");
        SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam, &s_stYolov3tinyModel);
        *model = NULL;
        return -1;
}

/*
 * 销毁Yolov2模型
 * Destroy Yolov2 model
 */
void Yolo2Destory(SAMPLE_SVP_NNIE_CFG_S *self)
{
    SampleSvpNnieYolov2Deinit(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam, &s_stYolov2Model);
    SAMPLE_COMM_SVP_CheckSysExit();
    free(self);
}

/*
 * 销毁Yolov3tiny模型
 * Destroy Yolov3tiny model
 */
void Yolo3tinyDestory(SAMPLE_SVP_NNIE_CFG_S *self)
{
    SAMPLE_SVP_NNIE_Yolov3tiny_Deinit(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam, &s_stYolov3tinyModel);
    SAMPLE_COMM_SVP_CheckSysExit();
    free(self);
}

/*
 * 获取结果
 * Fetch result
 */
static void Yolo2FetchRes(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum,
    DetectObjInfo resBuf[], int resSize, int* resLen)
{
    HI_U32 i;
    HI_U32 j;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias;
    HI_U32 u32BboxBias;
    HI_FLOAT f32Score;
    HI_S32* ps32Score = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstScore->u64VirAddr);
    HI_S32* ps32Roi = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstRoi->u64VirAddr);
    HI_S32* ps32ClassRoiNum = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstClassRoiNum->u64VirAddr);
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;

    HI_ASSERT(u32ClassNum == 2); // 2: the number of class
    HI_ASSERT(resSize > 0);
    int resId = 0;
    *resLen = 0;
    memset_s(resBuf, resSize * sizeof(resBuf[0]), 0x00, resSize * sizeof(resBuf[0]));

    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++) {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * SAMPLE_SVP_NNIE_COORDI_NUM;
        /*
         * 如果置信度分数大于结果阈值，则打印结果
         * If the confidence score greater than result threshold, the result will be printed
         */
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / SAMPLE_SVP_NNIE_QUANT_BASE >=
            THRESH_MIN && ps32ClassRoiNum[i] != 0) {
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++) {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / SAMPLE_SVP_NNIE_QUANT_BASE;
            if (f32Score < THRESH_MIN) {
                SAMPLE_PRT("f32Score:%.2f\n", f32Score);
                break;
            }
            if (resId >= resSize) {
                SAMPLE_PRT("yolo2 resBuf full\n");
                break;
            }
            resBuf[resId].cls = 1; // class 1
            resBuf[resId].score = f32Score;

            RectBox *box = &resBuf[resId].box;
            box->xmin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM];
            box->ymin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_1];
            box->xmax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_2];
            box->ymax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_3];
            if (box->xmin >= box->xmax || box->ymin >= box->ymax) {
                SAMPLE_PRT("yolo1_orig: {%d, %d, %d, %d}, %f, discard for coord ERR\n",
                    box->xmin, box->ymin, box->xmax, box->ymax, f32Score);
            } else {
                resId++;
            }
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }

    *resLen = resId;
}

static void Yolo3tinyFetchRes(SVP_BLOB_S *pstDstScore, SVP_BLOB_S *pstDstRoi, SVP_BLOB_S *pstClassRoiNum,
    DetectObjInfo resBuf[], int resSize, int* resLen)
{
    HI_U32 i;
    HI_U32 j;
    HI_U32 u32RoiNumBias = 0;
    HI_U32 u32ScoreBias;
    HI_U32 u32BboxBias;
    HI_FLOAT f32Score;
    HI_S32* ps32Score = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstScore->u64VirAddr);
    HI_S32* ps32Roi = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstDstRoi->u64VirAddr);
    HI_S32* ps32ClassRoiNum = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_S32, pstClassRoiNum->u64VirAddr);
    HI_U32 u32ClassNum = pstClassRoiNum->unShape.stWhc.u32Width;

    HI_ASSERT(u32ClassNum == 13); // 2: the number of class
    HI_ASSERT(resSize > 0);
    int resId = 0;
    *resLen = 0;
    memset_s(resBuf, resSize * sizeof(resBuf[0]), 0x00, resSize * sizeof(resBuf[0]));

    u32RoiNumBias += ps32ClassRoiNum[0];
    for (i = 1; i < u32ClassNum; i++) {
        u32ScoreBias = u32RoiNumBias;
        u32BboxBias = u32RoiNumBias * SAMPLE_SVP_NNIE_COORDI_NUM;
        /*
         * 如果置信度分数大于结果阈值，则打印结果
         * If the confidence score greater than result threshold, the result will be printed
         */
        if ((HI_FLOAT)ps32Score[u32ScoreBias] / SAMPLE_SVP_NNIE_QUANT_BASE >=
            THRESH_MIN && ps32ClassRoiNum[i] != 0) {
        }
        for (j = 0; j < (HI_U32)ps32ClassRoiNum[i]; j++) {
            f32Score = (HI_FLOAT)ps32Score[u32ScoreBias + j] / SAMPLE_SVP_NNIE_QUANT_BASE;
            if (f32Score < THRESH_MIN) {
                SAMPLE_PRT("f32Score:%.2f\n", f32Score);
                break;
            }
            if (resId >= resSize) {
                SAMPLE_PRT("yolo3 resBuf full\n");
                break;
            }
            resBuf[resId].cls = i;
            resBuf[resId].score = f32Score;

            RectBox *box = &resBuf[resId].box;
            box->xmin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM];
            box->ymin = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_1];
            box->xmax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_2];
            box->ymax = ps32Roi[u32BboxBias + j * SAMPLE_SVP_NNIE_COORDI_NUM + ARRAY_SUBSCRIPT_OFFSET_3];
            if (box->xmin >= box->xmax || box->ymin >= box->ymax) {
                SAMPLE_PRT("yolo3tiny_orig: num: %d {%d, %d, %d, %d}, %f, discard for coord ERR\n",
                    i, box->xmin, box->ymin, box->xmax, box->ymax, f32Score);
            } else {
                resId++;
            }
        }
        u32RoiNumBias += ps32ClassRoiNum[i];
    }

    *resLen = resId;
}

/*
 * Yolo2对一帧yuv图片进行推理
 * Calculation yuv image
 */
int Yolo2CalImg(SAMPLE_SVP_NNIE_CFG_S* self,
    const IVE_IMAGE_S *img, DetectObjInfo resBuf[], int resSize, int* resLen)
{
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
    HI_S32 s32Ret;

    /*
     * 填充源数据
     * Fill src data
     */
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;

    s32Ret = FillNnieByImg(self, &s_stYolov2NnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*
     * NNIE推理(process the 0-th segment)
     * NNIE process(process the 0-th segment)
     */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov2NnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /*
     * CPU处理
     * 如果用户更改了网络结构，请确保SAMPLE_SVP_NNIE_Yolov2_GetResult函数的输入数据是正确的
     *
     * Software process
     * if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov2_GetResult
     * function's input datas are correct
     */
    s32Ret = SAMPLE_SVP_NNIE_Yolov2_GetResult(&s_stYolov2NnieParam, &s_stYolov2SoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV2_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov2_GetResult failed!\n");

    Yolo2FetchRes(&s_stYolov2SoftwareParam.stDstScore,
        &s_stYolov2SoftwareParam.stDstRoi, &s_stYolov2SoftwareParam.stClassRoiNum, resBuf, resSize, resLen);
    return 0;

    YOLOV2_FAIL_0:
        return -1;
}

/*
 * Yolo3tiny对一帧yuv图片进行推理
 * Calculation yuv image
 */
int Yolo3tinyCalImg(SAMPLE_SVP_NNIE_CFG_S* self,
    const IVE_IMAGE_S *img, DetectObjInfo resBuf[], int resSize, int* resLen)
{
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
    HI_S32 s32Ret;
    HI_FLOAT f32PrintResultThresh = 0.5f;
    
    /*
     * 填充源数据
     * Fill src data
     */
    self->pszPic = NULL;
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    printf("In the FillNnieByImg\n");
    s32Ret = FillNnieByImg(self, &s_stYolov3tinyNnieParam, 0, 0, img);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");
    printf("Out of FillNnieByImg\n");
    /*
     * NNIE推理(process the 0-th segment)
     * NNIE process(process the 0-th segment)
     */
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stYolov3tinyNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");
    
    /*
     * CPU处理
     * 如果用户更改了网络结构，请确保SAMPLE_SVP_NNIE_Yolov3tiny_GetResult函数的输入数据是正确的
     *
     * Software process
     * if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Yolov3tiny_GetResult
     * function's input datas are correct
     */
    s32Ret = SAMPLE_SVP_NNIE_Yolov3tiny_GetResult(&s_stYolov3tinyNnieParam, &s_stYolov3tinySoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, YOLOV3TINY_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Yolov3tiny_GetResult failed!\n");

    Yolo3tinyFetchRes(&s_stYolov3tinySoftwareParam.stDstScore,
        &s_stYolov3tinySoftwareParam.stDstRoi, &s_stYolov3tinySoftwareParam.stClassRoiNum, resBuf, resSize, resLen);

    SAMPLE_SVP_TRACE_INFO("Yolov3tiny result:\n");
    (void)SAMPLE_SVP_NNIE_Detection_PrintResult(&s_stYolov3tinySoftwareParam.stDstScore, &s_stYolov3tinySoftwareParam.stDstRoi,
    &s_stYolov3tinySoftwareParam.stClassRoiNum, f32PrintResultThresh);

    return 0;



    YOLOV3TINY_FAIL_0:
        return -1;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
