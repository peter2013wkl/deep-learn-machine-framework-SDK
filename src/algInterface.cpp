#include "mvInterface.h"
#include "mvImageUtils.h"
#include "mvHeapMem.h"
#include "mvFeatureComp.h"
#include "mvMathUtils.h"
//sift
#include "surflib.h"
#include "sift_utils.h"
#include "mvXform.h"
#include "mvSurf.h"
#include "sift_imgfeatures.h"
#include "mvDrawUtils.h"
#include "mvImage.h"
#include "mvImageCCL.h"
#include "mvObjectMatch.h"
#include "mvIntergralImage.h"
#include "mvInterfaceUtils.h"  
#include "mvImagePreProcess.h"
#include "mvObjectDetRoiFilter.h"
#include "mvAlgSuanzi.h"
#include "mvDaemonWorker.h"


using namespace cv;

static int InsPerMit = 0;
static void *heapMem = NULL;
static unsigned int MV_SYS_MEMORY = MEMSIZE;

void mvAlgInitParamCreate(void *ppAlg)
{
	int i, j;
	algHandle *pAlg = (algHandle*)ppAlg;
	mvResult *pmvRes = &pAlg->result;

	//pmvRes = (mvResult*)mvMalloc(sizeof(mvResult));
	if (pmvRes)
	{
		pmvRes->matObjs = pAlg->matObjList;
		pmvRes->tmpObjs = pAlg->tempObjs;
		//pmvRes->matObjs = NULL;
		pmvRes->reserve = 0;
	}
	switch (pAlg->type)
	{
	case MV_ALG_FEATURE_TMP:
		mvTempLocParamCreate(ppAlg);
		break;
	case MV_ALG_ZJLOC_DET:
		mvLocAndMatchInitParamCreate(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET1:
		mvZMCothDet1InitParamCreate(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET2:
		mvZMCothDet2InitParamCreate(ppAlg);
		break;
	case MV_ALG_LZCOUNTER:
		mvLZCounterInitParamCreate(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_DL1:
		mvZMCothDetDL_InitParamCreate(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_CAFFE_SSD:
		mvZMCothDetDL_InitParamCreate(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_DL2:
	case MV_ALG_ZMCLOTH_DET_DL3:
	case MV_ALG_ZMCLOTH_DET_CAFFE_FCN:
		mvZMCothDetDL_FCN_InitParamCreate(ppAlg);
		break;
	default:
		break;;
	}

	return;
}

void mvAlgResultDestroy(void *ppAlg)
{
	int i, j;
	algHandle *pAlg = (algHandle*)ppAlg;
	mvResult *pmvRes = &pAlg->result;

	switch (pAlg->type)
	{
	case MV_ALG_FEATURE_TMP:
		mvTempLocParamDestroy(ppAlg);
		break;
	case MV_ALG_ZJLOC_DET:
		mvLocAndMatchInitParamDestroy(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET1:
		mvZJSurfaceDetResultDestroy(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET2:
		mvZMColthDet2InitParamDestroy(ppAlg);
		break;
	case MV_ALG_LZCOUNTER:
		mvLZCounterInitParamDestroy(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_DL1:
		mvZMColthDetDL_InitParamDestroy(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_CAFFE_SSD:
		mvZMColthDetDL_InitParamDestroy(ppAlg);
		break;
	case MV_ALG_ZMCLOTH_DET_DL2:
	case MV_ALG_ZMCLOTH_DET_DL3:
		mvZMColthDetDL_FCN_InitParamDestroy(ppAlg);
		break;
	default:
		break;;
	}

	return;
}

int mvAlgInit(void *ppAlg, int width, int height, initParam *para)
{
	algHandle *pAlg = (algHandle*)ppAlg;
	CvSize dstAlg;;
	int ret, i;
	mvEngineCfg *pa;
	char strMac[200];
	int macLen;
	mvPoint seed;
	char path[512];

#ifdef LIC_CHECK
	if (!mvLicValidate(pAlg->licNum))
		return 0;  // 
#endif

	mvHeapMonitorReset();
	//pAlg
	mvLog("begin to load config file...\n");

	if ((ret = mvAlgParamConfig(ppAlg, width, height, para)) != MV_OK)
	{
		mvLog("load config file error!\n");
		return ret;
	}

	mvLog("begin to create image buffer[%d x %d] ...\n", pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight);

	pAlg->pGrImg = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);  //alg image
	if (pAlg->pGrImg == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->pGrImg");

	pAlg->tmpGray = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);  //alg image
	if (pAlg->tmpGray == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->tmpGray");

	//if (pAlg->runcfg.algflag)
	//{
	//	pAlg->pLocGrImg = cvCreateImage(cvSize(pAlg->runcfg.locWidth, pAlg->runcfg.locHeight), IPL_DEPTH_8U, 1);  //alg image
	//	if (pAlg->pLocGrImg == NULL)
	//		return MV_ERROR;
	//	mvHeapMonitorIncrease("pAlg->pLocGrImg");
	//}

	pAlg->pCCLImage = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);
	if (pAlg->pCCLImage == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->pCCLImage");

	pAlg->pBinImg = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);
	if (pAlg->pBinImg == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->pBinImg");

	pAlg->pEdgeImg = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);
	if (pAlg->pEdgeImg == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->pEdgeImg");

	pAlg->pSEdgeImg = cvCreateImage(cvSize(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight), IPL_DEPTH_8U, 1);
	if (pAlg->pSEdgeImg == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->pSEdgeImg");

	pAlg->intImage = mvImageIntCreate(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight);
	if (pAlg->intImage == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->intImage");

	pAlg->runcfg.detRoi.roiMap = mvImageCreate2(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight);
	if (pAlg->runcfg.detRoi.roiMap == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->runcfg.detRoi.roiMap");

	pAlg->objOriginal.objDis = (mvObject*)mvMalloc(sizeof(mvObject)*MAX_DST_OBJ);
	if (pAlg->objOriginal.objDis == NULL)
		return MV_ERROR;
	mvMemset(pAlg->objOriginal.objDis, 0, sizeof(mvObject)*MAX_DST_OBJ);
	mvHeapMonitorIncrease("pAlg->objOriginal.objDis");

	pAlg->matObjList.matObj = (matchObj*)mvMalloc(sizeof(matchObj)*MAX_MATCH_OBJ);
	if (pAlg->matObjList.matObj == NULL)
		return MV_ERROR;
	mvMemset(pAlg->matObjList.matObj, 0, sizeof(matchObj)*MAX_DST_OBJ);
	mvHeapMonitorIncrease("pAlg->matObjList.matObj");

	pAlg->tempObjs.tmpObjs = (tempObj*)mvMalloc(sizeof(tempObj)*MAX_TEMP_OBJ);
	if (pAlg->tempObjs.tmpObjs == NULL)
		return MV_ERROR;
	mvMemset(pAlg->tempObjs.tmpObjs, 0, sizeof(tempObj)*MAX_TEMP_OBJ);
	mvHeapMonitorIncrease("pAlg->tempObjs.tmpObjs");
#ifdef MV_USE_FEAT_DESC
	mvSurf *opr;
	/* not implenmented "new", so we use our's mvMalloc */
	pAlg->featDesc = (void*)mvMalloc(sizeof(mvSurf));
	if (pAlg->featDesc == NULL)
		return MV_ERROR;
	opr = (mvSurf*)(pAlg->featDesc);
	opr->init();
	mvHeapMonitorIncrease("pAlg->featDesc");
#endif

#ifdef MV_CMT_TRACKER
	myCMT *cmt;
	/* not implenmented "new", so we use our's mvMalloc */
	pAlg->cmtTracker = (void*)new myCMT;
	if (pAlg->cmtTracker == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("mvCmt");
#endif

#ifdef MV_USE_FEATSBANK
	//for featset, not the tmpFeatSet
	if ((pAlg->featsBank = mvCreateFeatsBank(pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight)) == NULL)
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->featsBank");
#endif

#ifdef MV_USE_TRACKER
	mvObjectTrackerInit(&pAlg->objTraker, pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight);
	mvHeapMonitorIncrease("pAlg->objTraker");
#endif

	mvLog("begin to load tmpl.dat...\n");
	ret = mvTemplateRead(pAlg, &pAlg->feaComp.tmFeatSet, &pAlg->tempObjs, &pAlg->objTemplate);
	if (!ret)
	{
		mvLog("load template file failed.\n");
		return MV_ERROR;
	}

	//if (pAlg->type == MV_ALG_SHAPE_MATCH
	//	|| pAlg->type == MV_ALG_LOC_MATCH
	//	|| pAlg->type == MV_ALG_LOC_TMP_LOC)
	{
		mvLog("begin to init detRoi...\n");
		mvAlgInitDetRoi(ppAlg);
	}
#ifdef MV_USE_CCL
	//create ccl components!
	if ((ret = mvCCLItemsCreate(&pAlg->cclOrg, pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight))) //create ccl
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->cclOrg");

	//create ccl components!
	if ((ret = mvCCLItemsCreate(&pAlg->cclTmp, pAlg->runcfg.sclWidth, pAlg->runcfg.sclHeight))) //create ccl
		return MV_ERROR;
	mvHeapMonitorIncrease("pAlg->cclTmp");
#endif
	//create mvResult struct
	mvAlgInitParamCreate(pAlg);
	mvHeapMonitorIncrease("pAlg->mvResult");
	pAlg->imgCr.width = pAlg->runcfg.orgWidth;
	pAlg->imgCr.height = pAlg->runcfg.orgHeight;
	pAlg->imgCr.imageSize = 0;

	mvLog("alg instance init ok.\n");

	return MV_OK;
}

void mvSystemDeleteIns(void);

void mvAlgDelete(void *ppAlg)
{
	int i;
	algHandle *pAlg = (algHandle*)ppAlg;
	void *heapMem = NULL;
	//
	if (pAlg == NULL)
		return;

	while (pAlg->runStatus == MVALG_RUNNING);

	if (pAlg->objOriginal.objDis)
	{
		mvFree(pAlg->objOriginal.objDis);
		mvHeapMonitorDecrease("pAlg->objOriginal.objDis");
	}
	if (pAlg->matObjList.matObj)
	{
		mvFree(pAlg->matObjList.matObj);
		mvHeapMonitorDecrease("pAlg->matObjList.matObj");
	}
	if (pAlg->tempObjs.tmpObjs)
	{
		mvFree(pAlg->tempObjs.tmpObjs);
		mvHeapMonitorDecrease("pAlg->tempObjs.tmpObjs");
	}

	if (pAlg->pFrameAlg)
	{
		if (pAlg->pInputImage->width == pAlg->pFrameAlg->width
			&& pAlg->pInputImage->height == pAlg->pFrameAlg->height)
		{//not resize
			cvReleaseImageHeader(&pAlg->pFrameAlg);
		}
		else
			cvReleaseImage(&pAlg->pFrameAlg);

		mvHeapMonitorDecrease("pAlg->pFrameAlg");

	}

	if (pAlg->pInputImage)
	{
		if (pAlg->runcfg.channel > 3)
			cvReleaseImage(&pAlg->pInputImage);
		else
			cvReleaseImageHeader(&pAlg->pInputImage);

		mvHeapMonitorDecrease("pAlg->pInputImage");
	}
	if (pAlg->pGrImg)
	{
		cvReleaseImage(&pAlg->pGrImg);
		mvHeapMonitorDecrease("pAlg->pGrImg");
	}
	//if (pAlg->pLocGrImg)
	//{
	//	cvReleaseImage(&pAlg->pLocGrImg);
	//	mvHeapMonitorDecrease("pAlg->pLocGrImg");
	//}
	if (pAlg->pCCLImage)
	{
		cvReleaseImage(&pAlg->pCCLImage);
		mvHeapMonitorDecrease("pAlg->pCCLImage");
	}
	if (pAlg->pBinImg)
	{
		cvReleaseImage(&pAlg->pBinImg);
		mvHeapMonitorDecrease("pAlg->pBinImg");
	}
	//if (pAlg->pYellowBin)
	//{
	//	cvReleaseImage(&pAlg->pYellowBin);
	//	mvHeapMonitorDecrease("pAlg->pYellowBin");
	//}
	if (pAlg->pEdgeImg)
	{
		cvReleaseImage(&pAlg->pEdgeImg);
		mvHeapMonitorDecrease("pAlg->pEdgeImg");
	}
	if (pAlg->pSEdgeImg)
	{
		cvReleaseImage(&pAlg->pSEdgeImg);
		mvHeapMonitorDecrease("pAlg->pSEdgeImg");
	}
	if (pAlg->intImage)
	{
		mvImageIntDelete(pAlg->intImage);
		mvHeapMonitorDecrease("pAlg->intImage");
	}
	if (pAlg->tmpIntImage)
	{
		mvImageIntDelete(pAlg->tmpIntImage);
		mvHeapMonitorDecrease("pAlg->tmpIntImage");
	}

	if (pAlg->tmpGray)
	{
		cvReleaseImage(&pAlg->tmpGray);
		//mvHeapMonitorDecrease("pAlg->tmpGray");
	}

	if (pAlg->tmpEdge)
	{
		cvReleaseImage(&pAlg->tmpEdge);
		mvHeapMonitorDecrease("pAlg->tmpEdge");
	}

	if (pAlg->tmpSEdge)
	{
		cvReleaseImage(&pAlg->tmpSEdge);
		mvHeapMonitorDecrease("pAlg->tmpSEdge");
	}

	if (pAlg->tmpImg)
	{
		cvReleaseImage(&pAlg->tmpImg);
		mvHeapMonitorDecrease("pAlg->tmpImg");
	}
	if (pAlg->runcfg.detRoi.roiMap)
	{
		mvImageDelete(pAlg->runcfg.detRoi.roiMap);
		mvHeapMonitorDecrease("pAlg->runcfg.detRoi.roiMap");
	}
#ifdef MV_USE_FEAT_DESC
	mvSurf *opr;

	/* not implenment the operator new, so we use our's mvMalloc */
	if (pAlg->featDesc)
	{
		opr = (mvSurf*)pAlg->featDesc;
		opr->uninit();
		mvFree(pAlg->featDesc);
		mvHeapMonitorDecrease("pAlg->featDesc");
	}
#endif

#ifdef MV_USE_FEATSBANK
	if (pAlg->featsBank)
	{
		mvFeatsBankDelete(pAlg->featsBank);
		mvHeapMonitorDecrease("pAlg->featsBank");
	}
#endif

#ifdef MV_CMT_TRACKER
	myCMT *cmt;
	/* not implenment of operator new, so we use our's mvMalloc */
	if (pAlg->cmtTracker)
	{
		cmt = (myCMT*)pAlg->cmtTracker;
		delete cmt;
		mvHeapMonitorDecrease("mvCmt");
	}
#endif

	mvTemplateDestroy(pAlg, &pAlg->feaComp.tmFeatSet, &pAlg->tempObjs, &pAlg->objTemplate);

#ifdef MV_USE_TRACKER
	mvObjectTrackerDestroy(&pAlg->objTraker);
	mvHeapMonitorDecrease("pAlg->objTraker");
#endif

	//delete ccl components!
	mvCCLItemsDestroy(&pAlg->cclOrg);
	mvHeapMonitorDecrease("pAlg->cclOrg");
	mvCCLItemsDestroy(&pAlg->cclTmp);
	mvHeapMonitorDecrease("pAlg->cclTmp");

	mvAlgResultDestroy(&pAlg->result);
	mvHeapMonitorDecrease("pAlg->mvResult");
	if (pAlg)
	{
		mvParamSaveToFile(pAlg);
		/* free user data*/
		if (pAlg->userdat2)
			mvFree(pAlg->userdat2);
		if (pAlg->userdat3)
			mvFree(pAlg->userdat3);
		//if (pAlg->userdat4)
		//	mvFree(pAlg->userdat4);

		mvFree(pAlg);
	}
	pAlg = NULL;

	if (InsPerMit)
		InsPerMit--;

	mvSystemDeleteIns();

	return;
}

void mvInstanceDelete(void *pAlg)
{

	mvAlgDelete(pAlg);

	return;
}

void mvSystemDeleteIns(void)
{
#ifdef MV_USE_DAEMON_THREAD
	if (InsPerMit == 0)
		DaemonWorkerStop();
#endif

#ifdef USE_MVHEAPMEM
	if (InsPerMit == 0 && heapMem)
	{
#ifdef MV_ENABLE_LOG
		mvLog("delete instance...\n");
		mvLogDelete();

#endif
		free(heapMem);
		heapMem = NULL;
	}
#else

	if (InsPerMit == 0)
	{
#ifdef MV_ENABLE_LOG
		mvLogDelete();
		mvLog("delete ins ok.\n");
#endif
	}
#endif
}

int mvSetSysMemory(int mbyte)
{
	MV_SYS_MEMORY = mbyte * 1024 * 1024;

	return MV_OK;
}

void* mvSystemAllocIns(int numIns, int width, int height, char *path)
{
	int ret = 0;
	unsigned int nsize;

#ifdef USE_MVHEAPMEM
	if (heapMem == NULL)
	{
		nsize = MV_SYS_MEMORY * numIns;
		heapMem = (void*)malloc(nsize);
		if (heapMem)
		{
			ret = mvHeapMemInit(heapMem, (void*)((unsigned char*)heapMem + nsize));
			if (!ret)
			{
				free(heapMem);
				heapMem = NULL;
			}
			InsPerMit = 0;
		}
		else
		{
			mvLog00(path, "iMV::system error,no enough memory!");
		}

#ifdef MV_ENABLE_LOG
		mvLogInit(path);
#endif
#ifdef MV_USE_DAEMON_THREAD
		DaemonWorkerStart();
#endif
	}
	return heapMem;
#else 
	return (void*)1;
#endif
}

void* mvInstanceAlloc(int width, int height, algType type, initParam *para)
{
	algHandle *pAlg = NULL;
	int ret;
	int ly, lm, ld;
	char path[512];

	//check Lic
#ifdef LIC_CHECK
	if (LIC_REG_FLAG == MV_ERROR)
		return NULL;
#endif
	if (para == NULL)
		strcpy(path, "./imvcfg/");
	else
		strcpy(path, para->cfgPath);

	if (mvSystemAllocIns(MAX_ALG_INSTANCE, width, height, path) == NULL)
		return NULL;

	if (InsPerMit < MAX_ALG_INSTANCE)
	{
		pAlg = (algHandle*)mvMalloc(sizeof(algHandle));

		if (pAlg == NULL)
		{
			mvLog("pAlg is NULL, no engough memory\n");
			return pAlg;
		}

		mvMemset(pAlg, 0, sizeof(algHandle));
		mvMemset(pAlg->runcfg.cfgPath, 0, 512);

		strcpy(pAlg->runcfg.cfgPath, path);
		pAlg->heapMem = (void*)heapMem;
		InsPerMit++;
	}
	//check Lic
#ifdef LIC_CHECK
	if (LIC_REG_FLAG == MV_ERROR)
	{
		pAlg = NULL;
		mvFree(pAlg);
	}
	strcpy(pAlg->licNum, LIC_NUM);
#endif

	//return NULL;
	if (pAlg)
	{
		pAlg->type = (algType)type;
		ret = mvAlgInit(pAlg, width, height, para);
		if (ret < MV_OK)
		{
			mvLog("pAlg init failed.\n");
			mvFree(pAlg);
			pAlg = NULL;
			return NULL;
		}

#ifdef MV_USE_DAEMON_THREAD
		pAlg->licStatus = DaemonGetLicStatusPoint();
#else
		pAlg->licStatus = DaemonGetLicStatusPoint();
#endif

	}

	return (void*)pAlg;
}

void mvAlgTempReset(void *ppAlg, initParam *para)
{
	algHandle *pAlg = (algHandle*)ppAlg;

	//if (pAlg->pLocGrImg)
	//{
	//	cvReleaseImage(&pAlg->pLocGrImg);
	//	mvHeapMonitorDecrease("pAlg->pLocGrImg");
	//}

	if (pAlg->tmpIntImage)
	{
		mvImageIntDelete(pAlg->tmpIntImage);
	}

	if (pAlg->tmpGray)
	{
		cvReleaseImage(&pAlg->tmpGray);
	}

	if (pAlg->tmpEdge)
	{
		cvReleaseImage(&pAlg->tmpEdge);
	}

	if (pAlg->tmpSEdge)
	{
		cvReleaseImage(&pAlg->tmpSEdge);
	}

	if (pAlg->tmpImg)
	{
		cvReleaseImage(&pAlg->tmpImg);
		mvHeapMonitorDecrease("pAlg->tmpImg");
	}

	mvAlgInitParamSet(pAlg, para);

	pAlg->initEngine = MV_FALSE;
}

//mvFPoint mvGetMapPoint(void *ppAlg, float x, float y, int type)
//{
//	algHandle *pAlg = (algHandle*)ppAlg;
//	mvFPoint fp;
//	mvPoint offset, offset2;
//	mvFeatSet *dstSet;
//
//	if(type)
//	{
//		dstSet  = &pAlg->feaComp.featSet;
//		offset  = pAlg->feaComp.featSet.offset;
//		offset2 = pAlg->feaComp.tmFeatSet.offset;
//	}
//	else 
//	{
//		dstSet  = &pAlg->feaComp.tmFeatSet;
//		offset  = pAlg->feaComp.tmFeatSet.offset;
//		offset2 = pAlg->feaComp.featSet.offset;
//	}
//
//	//fp = mvCalcMapPoint(dstSet, x-offset.x, y-offset.y);
//	//fp.x += offset2.x; fp.y += offset2.y;
//
//	fp = mvCalcMapPoint(dstSet, x, y);
//	
//
//	return fp;
//}
//
//mvDetRoi mvGetMapDetRoi(void *ppAlg, int type)
//{
//	algHandle *pAlg = (algHandle*)ppAlg;
//	mvDetRoi det;
//	mvPolygon *tmpPoly,*dstPoly;
//	mvFPoint pt;
//	int i, j;
//	float xx, yy;
//	mvPoint offset, offset2;
//	int flag = 0;
//	mvDetRoi tmpDetRoi = pAlg->runcfg.detRoi;
//	mvDetRoi *pDet = &tmpDetRoi;
//
//	det = pAlg->runcfg.detRoi;
//
//	for(j = 0; j < pDet->numPoly; j++)
//	{
//		tmpPoly = &pDet->polys[j];
//		dstPoly = &det.polys[j];
//		for (i = 0; i < tmpPoly->num; i++)
//		{//make sure the point is in the image, and be closed!
//		    pt = mvGetMapPoint(pAlg, tmpPoly->ppnts[i].x, tmpPoly->ppnts[i].y, type);
//
//			dstPoly->ppnts[i].x = pt.x; /* + .5*/
//			dstPoly->ppnts[i].y = pt.y; /* + .5*/
//			//if (!flag)
//			//{
//			//	det.rect.pntUpLft = det.rect.pntDnRgt = dstPoly->ppnts[i];
//			//	flag = MV_TRUE;
//			//}
//			//det.rect.pntUpLft.x = MV_MIN(det.rect.pntUpLft.x, dstPoly->ppnts[i].x);
//			//det.rect.pntUpLft.y = MV_MIN(det.rect.pntUpLft.y, dstPoly->ppnts[i].y);
//			//det.rect.pntDnRgt.x = MV_MAX(det.rect.pntDnRgt.x, dstPoly->ppnts[i].x);
//			//det.rect.pntDnRgt.y = MV_MAX(det.rect.pntDnRgt.y, dstPoly->ppnts[i].y);
//		}
//		pt = mvGetMapPoint(pAlg, tmpPoly->seed.x, tmpPoly->seed.y, type);
//		dstPoly->seed.x = pt.x; /* + .5*/
//		dstPoly->seed.y = pt.y; /* + .5*/
//		pt = mvGetMapPoint(pAlg, tmpPoly->rect.pntUpLft.x, tmpPoly->rect.pntUpLft.y, type);
//		dstPoly->rect.pntUpLft.x = pt.x; /* + .5*/
//		dstPoly->rect.pntUpLft.y = pt.y; /* + .5*/
//		pt = mvGetMapPoint(pAlg, tmpPoly->rect.pntDnRgt.x, tmpPoly->rect.pntDnRgt.y, type);
//		dstPoly->rect.pntDnRgt.x = pt.x; /* + .5*/
//		dstPoly->rect.pntDnRgt.y = pt.y; /* + .5*/
//		dstPoly->lable = tmpPoly->lable;
//		dstPoly->uc   = tmpPoly->uc;
//		dstPoly->num   = tmpPoly->num;
//	}
//	//calc detRoi maxRect
//	//xx = pAlg->detRoi.rect.pntUpLft.x - offset.x; yy = pAlg->detRoi.rect.pntUpLft.y - offset.y;
//	//pt = mvGetMapPoint(pAlg, xx, yy);
//	//det.rect.pntUpLft.x = pt.x; /* + .5*/
//	//det.rect.pntUpLft.y = pt.y; /* + .5*/
//	//xx = pAlg->detRoi.rect.pntDnRgt.x - offset.x; yy = pAlg->detRoi.rect.pntDnRgt.y - offset.y;
//	//pt = mvGetMapPoint(pAlg, xx, yy);
//	//det.rect.pntDnRgt.x = pt.x; /* + .5*/
//	//det.rect.pntDnRgt.y = pt.y; /* + .5*/
//	//if (pAlg->detRoi.roiMap)
//	//{
//	//	mvImageDelete(pAlg->detRoi.roiMap);
//	//}
//
//	return det;
//}

int mvMatchProcess(void *ppAlg, mvInputImage *pImgInput)
{
	algHandle *pAlg = (algHandle*)ppAlg;
	mvResult *mvRes = (mvResult*)&pAlg->result;
	mvEngineCfg *pa = &pAlg->algParam;

	int initFlag = MV_ERROR;
	mvImage bm;
	int ret = MV_ERROR;
	int i;

	pAlg->runStatus = MVALG_RUNNING;
	//mvPrintHeapMemInfo();

	mvImagePreProcess(ppAlg, pImgInput);
	//pAlg->runStatus = MVALG_IDLE;
	//return 0;

	//mvPrintHeapMemInfo();
#ifdef MV_USE_OPENCV_FUN
	//pa->edgeMaxThres = 230;
	cvCanny(pAlg->pGrImg, pAlg->pEdgeImg, pa->edgeMaxThres / 3, pa->edgeMaxThres, 3);
#else
	//cvCanny(pAlg->pGrImg, pAlg->pEdgeImg, 10, 150, 3);
	//void CannyBfs2(unsigned char *glDat, unsigned char *edgeDat, int width, int height, int lowThres, int highThres)
	CannyBfs2((unsigned char*)pAlg->pGrImg->imageData, (unsigned char*)pAlg->pEdgeImg->imageData, pAlg->pGrImg->width, pAlg->pGrImg->height, 50, 100);
	//cvShowImage("edge", pAlg->pEdgeImg);
	//cvWaitKey(0);
#endif

	bm.imageData = (unsigned char*)pAlg->pEdgeImg->imageData;
	bm.width = pAlg->pEdgeImg->width;
	bm.height = pAlg->pEdgeImg->height;
	bm.imageSize = pAlg->pEdgeImg->imageSize;

	mvDetRoiMapFilter(pAlg->runcfg.detRoi.roiMap, (unsigned char*)pAlg->pEdgeImg->imageData);
	if (!pAlg->algParam.useDilate && !pAlg->algParam.useErode)
		cvCopy(pAlg->pEdgeImg, pAlg->pSEdgeImg);

	if (pAlg->algParam.useDilate)
		cvDilate(pAlg->pEdgeImg, pAlg->pSEdgeImg, 0, 1);
	if (pAlg->algParam.useErode)
		cvErode(pAlg->pSEdgeImg, pAlg->pSEdgeImg, 0, 1);

	//cvCopy(pAlg->pEdgeImg, pAlg->pSEdgeImg);
	mvCCLItemsReset(&pAlg->cclOrg);

	pAlg->mImage.imageData = (unsigned char*)pAlg->pSEdgeImg->imageData;
	pAlg->mImage.height = pAlg->pEdgeImg->height;
	pAlg->mImage.width = pAlg->pEdgeImg->width;
	pAlg->mImage.imageSize = pAlg->pSEdgeImg->imageSize;

	mvImagePaintBorderLines(&pAlg->mImage, 0, 2);

	pAlg->imgCCl.imageData = (unsigned char*)pAlg->pCCLImage->imageData;
	pAlg->imgCCl.height = pAlg->pCCLImage->height;
	pAlg->imgCCl.width = pAlg->pCCLImage->width;
	pAlg->imgCCl.imageSize = pAlg->pCCLImage->imageSize;

	mvCCLProcess(pAlg, pAlg->mImage, &pAlg->cclOrg, &pAlg->imgCCl, (unsigned char*)pAlg->pEdgeImg->imageData, pAlg);
	if (pAlg->cclOrg.iCompsNu)
	{
		//IntegralBinImage(&pAlg->imgCCl, pAlg->intImage);
		IntegralGrayAndEdgeImage(pAlg->pGrImg, &pAlg->imgCCl, pAlg->intImage);
	}

	//mvPrintHeapMemInfo();

	mvCCLObjExtractProcess((void*)pAlg, pAlg->pFrameAlg, pAlg->pGrImg, pAlg->pCCLImage);
	//mvPrintHeapMemInfo();
	mvObjectMatchProcess(pAlg, &pAlg->objTemplate, &pAlg->objOriginal, &pAlg->matObjList);
	//mvPrintHeapMemInfo();
	if (pa->disLevel & (0x01 << 12) || (pa->disLevel & (0x01 << 13)))
		CCLFilterComponentsDrawRectangle(&pAlg->cclOrg, &pAlg->imgCCl, pAlg);
	//if (pa->disLevel & (0x01<<14)  && !pAlg->type)
	//{
	//	cvShowImage("blob-analysis", pAlg->pCCLImage);
	//	cvWaitKey(1);
	//}
	//mvPrintHeapMemInfo();
	if (mvRes != NULL && pAlg->licStatus)
	{
		mvRes->matObjs = pAlg->matObjList;
		mvRes->tmpObjs = pAlg->tempObjs;
		//mvMatchObjCopyContours(mvRes->matObjs, &pAlg->matObjList);
	}
	//else
	//{
	//MatchObjsDrawAndDisplay(pAlg, &pAlg->matObjList);
	//}
	//
	//mvPrintHeapMemInfo();
	mvObjDataDestroy(ppAlg, &pAlg->objOriginal);

	//mvCCLItemsDestroy(&pAlg->cclOrg);
	//memcpy(pImgInput->pFrame, pAlg->pFrameAlg->imageData, pAlg->pInputImage->imageSize);
	pAlg->imgCr.imageData = (unsigned char*)pAlg->pInputImage->imageData;

	//pAlg->imgBlobs.imageData = (unsigned char*)pAlg->pCCLImage->imageData;

	pAlg->runCounter++;

	pAlg->runStatus = MVALG_IDLE;
	//mvPrintHeapMemInfo();
	return MV_OK;

err_exit:
	return MV_ERROR;
}

int mvAlgProcess(void *ppAlg, mvInputImage *pImgInput)
{
	algHandle *pAlg = (algHandle*)ppAlg;


	switch (pAlg->type)
	{
	case MV_ALG_FEATURE_TMP:
		return mvTempLocProcess(ppAlg, pImgInput);
	case MV_ALG_ZJLOC_DET:
		return mvZJLocationProcess(ppAlg, pImgInput);
	case MV_ALG_ZMCLOTH_DET1:
		return mvZMClothDet1Process(ppAlg, pImgInput);
	case MV_ALG_ZMCLOTH_DET2:
		return mvZMClothDet2Process(ppAlg, pImgInput);
	case MV_ALG_LZCOUNTER:
		return mvLZCounterProcess(ppAlg, pImgInput);
	case MV_ALG_ZMCLOTH_DET_DL1:
		return mvZMClothDetDL_Process(ppAlg, pImgInput);
	case MV_ALG_ZMCLOTH_DET_CAFFE_SSD:
		return mvZMClothDetDL_CAFFE_SSD_Process(pAlg, pImgInput);
	case MV_ALG_ZMCLOTH_DET_DL2:
		return mvZMClothDetDL_FCN_Process(ppAlg, pImgInput, 1);
	case MV_ALG_ZMCLOTH_DET_DL3:
		return mvZMClothDetDL_FCN_Process(ppAlg, pImgInput, 1);
	case MV_ALG_ZMCLOTH_DET_CAFFE_FCN:
		return mvZMClothDetDL_CAFFE_FCN_Process(ppAlg, pImgInput, 1);
	default:
		return MV_UNKNOWN_TYPE;
	}
}

int mvCheckVersion(int nsize)
{//align is important for 32bit or 64bit!!!!
	int n;
	int ret;

	if (sizeof(void*) == 4)
	{//由于matchObj不是整数倍， 所需要补齐,以下为手工添加：16
#ifdef USE_POLYGON_TYPE
		ret = sizeof(mvVersionHeader) - (sizeof(unMObjHeader) - sizeof(void*) + 12) + sizeof(void*);
#else
		ret = sizeof(mvVersionHeader) - (sizeof(unMObjHeader) - sizeof(void*) + 16) + sizeof(void*);
#endif
	}
	else
	{
		ret = sizeof(mvVersionHeader) - (sizeof(unMObjHeader)) + sizeof(void*);
	}
	//n = sizeof (matchObj);
	//n = sizeof (unMObjHeader);
	if (nsize != ret)
		return MV_ERROR;

	return MV_OK;
}



#ifdef _WIN32
#include <windows.h>
static HINSTANCE mvDL_CAFFE_Handle = NULL;
#endif

typedef cv::Mat(*mvDLCAFFE_process)(void *ppAlg, cv::Mat &frame, int flag);


static mvDLCAFFE_process       _tmvDLCAFFE_process = NULL;

static int dlrefcnt = 0;


cv::Mat FDLLmvDLCAFFE_process(void *ppAlg, cv::Mat &frame, int algtype)
{
	return (_tmvDLCAFFE_process(ppAlg, frame, algtype));
}


int mvDL_CAFFE_Model_Load(void)
{

	if (dlrefcnt == 0)
	{

		if (sizeof(void*) == 4)
			mvDL_CAFFE_Handle = ::LoadLibrary("mvDLModel.dll");
		else
			mvDL_CAFFE_Handle = ::LoadLibrary("mvDLModel.dll");

		if (mvDL_CAFFE_Handle == 0)
		{
			printf("Load mvCAFFE_DL.dll error.\n");

			return 0;
		}

		_tmvDLCAFFE_process = (mvDLCAFFE_process)::GetProcAddress(mvDL_CAFFE_Handle, "mvDLCAFFE_process");


		if (!_tmvDLCAFFE_process)
		{
			printf("mvCAFFE_DL.dll Load functions error!\n");
			return 1;
		}

		dlrefcnt++;

		return 1;
	}

	dlrefcnt++;

	return 1;
}


void mvDL_CAFFE_Model_Free(void)
{
	if (dlrefcnt)
		dlrefcnt--;

	if (_tmvDLCAFFE_process == 0)
	{
		_tmvDLCAFFE_process = NULL;

	}
}
