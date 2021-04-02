/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>

#include <math.h>

#include <stdio.h>
#include <string.h>
#include "cuda_runtime_api.h"

#include <opencv2/objdetect/objdetect.hpp>

#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include "nvdsinfer_custom_impl.h"
#include "nvds_version.h"


#include <iostream>
#include <vector>

#define NVINFER_PLUGIN "nvinfer"
#define NVINFERSERVER_PLUGIN "nvinferserver"

#define INFER_PGIE_CONFIG_FILE  "/opt/nvidia/deepstream/deepstream-5.0/sources/objectDetector_Yolo/dstest2_pgie_config.txt"
#define INFER_SGIE1_CONFIG_FILE "./dstest2_sgie1_config.txt"
#define INFER_SGIE2_CONFIG_FILE "dstensor_sgie2_config.txt"
#define INFER_SGIE3_CONFIG_FILE "dstensor_sgie3_config.txt"
#define INFERSERVER_PGIE_CONFIG_FILE  "inferserver/dstensor_pgie_config.txt"
#define INFERSERVER_SGIE1_CONFIG_FILE "inferserver/dstensor_sgie1_config.txt"
#define INFERSERVER_SGIE2_CONFIG_FILE "inferserver/dstensor_sgie2_config.txt"
#define INFERSERVER_SGIE3_CONFIG_FILE "inferserver/dstensor_sgie3_config.txt"

#define MAX_DISPLAY_LEN 64

#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

#define PGIE_DETECTED_CLASS_NUM 4

#define OSD_PROCESS_MODE 0

/* By default, OSD will not display text. To display text, change this to 1 */
#define OSD_DISPLAY_TEXT 0


#define TILED_OUTPUT_WIDTH 1280
#define TILED_OUTPUT_HEIGHT 720

/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* The muxer output resolution must be set if the input streams will be of
 * different resolution. The muxer will scale all the input frames to this
 * resolution. */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080


#define PGIE_NET_WIDTH 416
#define PGIE_NET_HEIGHT 416

/* Muxer batch formation timeout, for e.g. 40 millisec. Should ideally be set
 * based on the fastest source's framerate. */
#define MUXER_BATCH_TIMEOUT_USEC 40000

gint frame_number = 0;
/* These are the strings of the labels for the respective models */
const gchar sgie1_classes_str[12][32] = {
  "black", "blue", "brown", "gold", "green",
  "grey", "maroon", "orange", "red", "silver", "white", "yellow"
};

const gchar sgie2_classes_str[20][32] = {
  "Acura", "Audi", "BMW", "Chevrolet", "Chrysler",
  "Dodge", "Ford", "GMC", "Honda", "Hyundai", "Infiniti", "Jeep", "Kia",
  "Lexus", "Mazda", "Mercedes", "Nissan",
  "Subaru", "Toyota", "Volkswagen"
};

const gchar sgie3_classes_str[6][32] = {
  "coupe", "largevehicle", "sedan", "suv",
  "truck", "van"
};

const gchar pgie_classes_str[PGIE_DETECTED_CLASS_NUM][32] =
    { "Vehicle", "TwoWheeler", "Person", "RoadSign" };

/* gie_unique_id is one of the properties in the above dstensor_sgiex_config.txt
 * files. These should be unique and known when we want to parse the Metadata
 * respective to the sgie labels. Ideally these should be read from the config
 * files but for brevity we ensure they are same. */

const guint sgie1_unique_id = 2;
const guint sgie2_unique_id = 3;
const guint sgie3_unique_id = 4;

/* nvds_lib_major_version and nvds_lib_minor_version is the version number of
 * deepstream sdk */

unsigned int nvds_lib_major_version = NVDS_VERSION_MAJOR;
unsigned int nvds_lib_minor_version = NVDS_VERSION_MINOR;

/* This is the buffer probe function that we have registered on the sink pad
 * of the OSD element. All the infer elements in the pipeline shall attach
 * their metadata to the GstBuffer, here we will iterate & process the metadata
 * forex: class ids to strings, counting of class_id objects etc. */
static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
  GstBuffer *buf = (GstBuffer *) info->data;
  guint num_rects = 0;
  NvDsObjectMeta *obj_meta = NULL;
  guint vehicle_count = 0;
  guint person_count = 0;
  NvDsMetaList *l_frame = NULL;
  NvDsMetaList *l_obj = NULL;
  NvDsDisplayMeta *display_meta = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    int offset = 0;
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next) {
      obj_meta = (NvDsObjectMeta *) (l_obj->data);
      if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
        vehicle_count++;
        num_rects++;
      }
      if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
        person_count++;
        num_rects++;
      }
    }
    display_meta = nvds_acquire_display_meta_from_pool (batch_meta);
    NvOSD_TextParams *txt_params = &display_meta->text_params[0];
    display_meta->num_labels = 1;
    txt_params->display_text = (gchar *) g_malloc0 (MAX_DISPLAY_LEN);
    offset =
        snprintf (txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ",
        person_count);
    offset =
        snprintf (txt_params->display_text + offset, MAX_DISPLAY_LEN,
        "Vehicle = %d ", vehicle_count);

    /* Now set the offsets where the string should appear */
    txt_params->x_offset = 10;
    txt_params->y_offset = 12;

    /* Font , font-color and font-size */
    txt_params->font_params.font_name = (gchar *) "Serif";
    txt_params->font_params.font_size = 10;
    txt_params->font_params.font_color.red = 1.0;
    txt_params->font_params.font_color.green = 1.0;
    txt_params->font_params.font_color.blue = 1.0;
    txt_params->font_params.font_color.alpha = 1.0;

    /* Text background color */
    txt_params->set_bg_clr = 1;
    txt_params->text_bg_clr.red = 0.0;
    txt_params->text_bg_clr.green = 0.0;
    txt_params->text_bg_clr.blue = 0.0;
    txt_params->text_bg_clr.alpha = 1.0;

    nvds_add_display_meta_to_frame (frame_meta, display_meta);
  }


  //g_print ("Frame Number = %d Number of objects = %d "
      //"Vehicle Count = %d Person Count = %d\n",
      //frame_number, num_rects, vehicle_count, person_count);
  frame_number++;
  return GST_PAD_PROBE_OK;
}

//extern "C"
    //bool NvDsInferParseCustomResnet (std::vector < NvDsInferLayerInfo >
    //const &outputLayersInfo, NvDsInferNetworkInfo const &networkInfo,
    //NvDsInferParseDetectionParams const &detectionParams,
    //std::vector < NvDsInferObjectDetectionInfo > &objectList);

/* This is the buffer probe function that we have registered on the src pad
 * of the PGIE's next queue element. PGIE element in the pipeline shall attach
 * its NvDsInferTensorMeta to each frame metadata on GstBuffer, here we will
 * iterate & parse the tensor data to get detection bounding boxes. The result
 * would be attached as object-meta(NvDsObjectMeta) into the same frame metadata.
 */
static GstPadProbeReturn
pgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;
  static NvDsInferNetworkInfo networkInfo
  {
  PGIE_NET_WIDTH, PGIE_NET_HEIGHT, 3};
  NvDsInferParseDetectionParams detectionParams;
  detectionParams.numClassesConfigured = 1;
  detectionParams.perClassPreclusterThreshold = {0.2, 0.2, 0.2, 0.2};
  static float groupThreshold = 1;
  static float groupEps = 0.2;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Iterate user metadata in frames to search PGIE's tensor metadata */
    for (NvDsMetaList * l_user = frame_meta->frame_user_meta_list;
        l_user != NULL; l_user = l_user->next) {
      NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
      if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
        continue;

      /* convert to tensor metadata */
      NvDsInferTensorMeta *meta =
          (NvDsInferTensorMeta *) user_meta->user_meta_data;
      for (unsigned int i = 0; i < meta->num_output_layers; i++) {
        NvDsInferLayerInfo *info = &meta->output_layers_info[i];
        info->buffer = meta->out_buf_ptrs_host[i];
        if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
          cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
              info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
        }
      }
      /* Parse output tensor and fill detection results into objectList. */
      std::vector < NvDsInferLayerInfo >
          outputLayersInfo (meta->output_layers_info,
          meta->output_layers_info + meta->num_output_layers);
      std::vector < NvDsInferObjectDetectionInfo > objectList;
#if NVDS_VERSION_MAJOR >= 5
      if (nvds_lib_major_version >= 5) {
        if (meta->network_info.width != networkInfo.width ||
            meta->network_info.height != networkInfo.height ||
            meta->network_info.channels != networkInfo.channels) {
          g_error ("failed to check pgie network info\n");
        }
      }
#endif
      ///NvDsInferParseCustomResnet (outputLayersInfo, networkInfo,
          ///detectionParams, objectList);

      /* Seperate detection rectangles per class for grouping. */
      //std::vector < std::vector <
          //cv::Rect >> objectListClasses (PGIE_DETECTED_CLASS_NUM);
    //for (auto & obj:objectList) {
        //objectListClasses[obj.classId].emplace_back (obj.left, obj.top,
            //obj.width, obj.height);
      //}

      //for (uint32_t c = 0; c < objectListClasses.size (); ++c) {
        //auto & objlist = objectListClasses[c];
        //if (objlist.empty ())
          //continue;

        /* Merge and cluster similar detection results */
        //cv::groupRectangles (objlist, groupThreshold, groupEps);

        /* Iterate final rectangules and attach result into frame's obj_meta_list. */
      //for (const auto & rect:objlist) {
          //NvDsObjectMeta *obj_meta =
              //nvds_acquire_obj_meta_from_pool (batch_meta);
          //obj_meta->unique_component_id = meta->unique_id;
          //obj_meta->confidence = 0.0;

          /* This is an untracked object. Set tracking_id to -1. */
          //obj_meta->object_id = UNTRACKED_OBJECT_ID;
          //obj_meta->class_id = c;

          //NvOSD_RectParams & rect_params = obj_meta->rect_params;
          //NvOSD_TextParams & text_params = obj_meta->text_params;

          /* Assign bounding box coordinates. */
          //rect_params.left = rect.x * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
          //rect_params.top = rect.y * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;
          //rect_params.width = rect.width * MUXER_OUTPUT_WIDTH / PGIE_NET_WIDTH;
          ///rect_params.height =
              //rect.height * MUXER_OUTPUT_HEIGHT / PGIE_NET_HEIGHT;

          /* Border of width 3. */
          //rect_params.border_width = 3;
          //rect_params.has_bg_color = 0;
          //rect_params.border_color = (NvOSD_ColorParams) {
          //1, 0, 0, 1};

          /* display_text requires heap allocated memory. */
          //text_params.display_text = g_strdup (pgie_classes_str[c]);
          /* Display text above the left top corner of the object. */
          //text_params.x_offset = rect_params.left;
          //text_params.y_offset = rect_params.top - 10;
          /* Set black background for the text. */
          //text_params.set_bg_clr = 1;
          //text_params.text_bg_clr = (NvOSD_ColorParams) {
          //0, 0, 0, 1};
          /* Font face, size and color. */
          //text_params.font_params.font_name = (gchar *) "Serif";
          //text_params.font_params.font_size = 11;
          //text_params.font_params.font_color = (NvOSD_ColorParams) {
          //1, 1, 1, 1};
          //nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
       // }
      //}
    }
  }
  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

/* This is the buffer probe function that we have registered on the sink pad
 * of the tiler element. All SGIE infer elements in the pipeline shall attach
 * their NvDsInferTensorMeta to each object's metadata of each frame, here we will
 * iterate & parse the tensor data to get classification confidence and labels.
 * The result would be attached as classifier_meta into its object's metadata.
 */
static GstPadProbeReturn
sgie_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data)
{
  static guint use_device_mem = 0;

  NvDsBatchMeta *batch_meta =
      gst_buffer_get_nvds_batch_meta (GST_BUFFER (info->data));

  /* Iterate each frame metadata in batch */
  for (NvDsMetaList * l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) l_frame->data;

    /* Iterate object metadata in frame */
    for (NvDsMetaList * l_obj = frame_meta->obj_meta_list; l_obj != NULL;
        l_obj = l_obj->next) {
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) l_obj->data;

      /* Iterate user metadata in object to search SGIE's tensor data */
      for (NvDsMetaList * l_user = obj_meta->obj_user_meta_list; l_user != NULL;
          l_user = l_user->next) {
        NvDsUserMeta *user_meta = (NvDsUserMeta *) l_user->data;
        if (user_meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META)
          continue;

        /* convert to tensor metadata */
        NvDsInferTensorMeta *meta =
            (NvDsInferTensorMeta *) user_meta->user_meta_data;
        //std::cout<<"size "<< *(unsigned int *)meta->num_output_layers;
        for (unsigned int i = 0; i < meta->num_output_layers; i++) {
          NvDsInferLayerInfo *info = &meta->output_layers_info[i];
          
          info->buffer = meta->out_buf_ptrs_host[i];
          float (*array)[130] = (float (*)[130]) info->buffer;

          std::cout<<"Shape "<<info->inferDims.numElements<<std::endl;
          std::cout<<"128d Tensor"<<std::endl;
          for(int m =0;m<info->inferDims.numElements;m++){
          std::cout<<" "<< (*array)[m];
          }
          //double* ptr = (double*)meta->out_buf_ptrs_host[0];  // output layer 0
          //for( size_t i=0; i<info->inferDims.numElements; i++ )
          //{
            //std::cout << "Tensor " << i << ": " << ptr[i] << std::endl;
            //std::cout  << i << ": " << ptr[i];
           // }
          
          if (use_device_mem && meta->out_buf_ptrs_dev[i]) {
            cudaMemcpy (meta->out_buf_ptrs_host[i], meta->out_buf_ptrs_dev[i],
                info->inferDims.numElements * 4, cudaMemcpyDeviceToHost);
          }
          
        }
          
        //NvDsInferDimsCHW dims;

        //getDimsCHWFromDims (dims, meta->output_layers_info[0].inferDims);
        //unsigned int numClasses = dims.c;
        //float *outputCoverageBuffer = 
            //(float *) meta->output_layers_info[0].buffer;
        
        //std::cout<<"gxxgsss"<<*(float *)meta->output_layers_info[0].buffer;
        //float maxProbability = 0;
        //bool attrFound = false;
        //NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        ///for (unsigned int c = 0; c < numClasses; c++) {
          ///float probability = outputCoverageBuffer[c];
          ///if (probability > 0.51 && probability > maxProbability) {
            ///maxProbability = probability;
            ///attrFound = true;
            ///attr.attributeIndex = 0;
            ///attr.attributeValue = c;
            ///attr.attributeConfidence = probability;
          ///}
        ///}

        /* Generate classifer metadata and attach to obj_meta */
        ///if (attrFound) {
          ///NvDsClassifierMeta *classifier_meta =
              ///nvds_acquire_classifier_meta_from_pool (batch_meta);

          ///classifier_meta->unique_component_id = meta->unique_id;

          ///NvDsLabelInfo *label_info =
             /// nvds_acquire_label_info_meta_from_pool (batch_meta);
          ///label_info->result_class_id = attr.attributeValue;
          ///label_info->result_prob = attr.attributeConfidence;

          /* Fill label name */
          ///switch (meta->unique_id) {
            ///case sgie1_unique_id:
              ///strcpy (label_info->result_label,
                  ///sgie1_classes_str[label_info->result_class_id]);
              ///break;
            ///case sgie2_unique_id:
              ///strcpy (label_info->result_label,
                  ///sgie2_classes_str[label_info->result_class_id]);
              ///break;
            ///case sgie3_unique_id:
              ///strcpy (label_info->result_label,
                  ///sgie3_classes_str[label_info->result_class_id]);
              ///break;
            ///default:
              ///break;
          ///}
          ///gchar *temp = obj_meta->text_params.display_text;
          ///obj_meta->text_params.display_text =
              ///g_strconcat (temp, " ", label_info->result_label, nullptr);
          ///g_free (temp);

          ///nvds_add_label_info_meta_to_classifier (classifier_meta, label_info);
          ///nvds_add_classifier_meta_to_object (obj_meta, classifier_meta);
        ///}
      }
     
    }
  
  }

  use_device_mem = 1 - use_device_mem;
  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
#ifndef PLATFORM_TEGRA
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      break;
    }
#endif
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}
int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL,
    *sgie1, *queue1, *queue2, *queue3, *queue4, *queue5, *queue6, *nvvidconv = NULL,
      *nvosd = NULL, *tiler = NULL;

#ifdef PLATFORM_TEGRA
  GstElement *transform = NULL;
#endif
  GstBus *bus = NULL;
  guint bus_watch_id = 0;
  GstPad *osd_sink_pad = NULL, *queue_src_pad = NULL, *tiler_sink_pad = NULL, *tiler_src_pad = NULL;
  guint i, num_sources;
  guint tiler_rows, tiler_columns;
  guint pgie_batch_size;
  std::vector<std::string> files;
  gboolean is_nvinfer_server = FALSE;
  const char *infer_plugin = NVINFER_PLUGIN;

  nvds_version(&nvds_lib_major_version, &nvds_lib_minor_version);

    /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <uri1> [uri2] ... [uriN] \n", argv[0]);
    return -1;
  }
  num_sources = argc - 1;


  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */

  /* Create Pipeline element that will be a container of other elements */
  pipeline = gst_pipeline_new ("dstensor-pipeline");


  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add (GST_BIN (pipeline), streammux);


    for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };
    GstElement *source_bin = create_source_bin (i, argv[i + 1]);

    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);
  }

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make (infer_plugin, "primary-nvinference-engine");

  /* Add queue elements between every two elements */
  //queue = gst_element_factory_make ("queue", "queue");
  queue1 = gst_element_factory_make ("queue", "queue1");
  queue2 = gst_element_factory_make ("queue", "queue2");
  queue3 = gst_element_factory_make ("queue", "queue3");
  queue4 = gst_element_factory_make ("queue", "queue4");
  queue5 = gst_element_factory_make ("queue", "queue5");
  queue6 = gst_element_factory_make ("queue", "queue6");


  /* We need three secondary gies so lets create 3 more instances of
     nvinfer */
  sgie1 = gst_element_factory_make (infer_plugin, "secondary1-nvinference-engine");

  /* Use nvtiler to composite the batched frames into a 2D tiled array based
   * on the source of the frames. */
  tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
#ifdef PLATFORM_TEGRA
  //transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  transform = gst_element_factory_make ("queue", "queue");
#endif
  //sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
  sink = gst_element_factory_make ("nvoverlaysink", "nvvideo-renderer");

  if (!pgie || !tiler || !nvvidconv || !nvosd || !sink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

#ifdef PLATFORM_TEGRA
  if(!transform) {
    g_printerr ("One tegra element could not be created. Exiting.\n");
    return -1;
  }
#endif

  g_object_set (G_OBJECT (streammux), "batch-size", num_sources, NULL);
  g_object_set (G_OBJECT(sink), "sync", FALSE, NULL);

  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height",
      MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* Configure the nvinfer element using the nvinfer config file. */
  //g_object_set (G_OBJECT (pgie),
      //"config-file-path", "dstest3_pgie_config.txt", NULL);

  /* Override the batch-size set in the config file with the number of sources. */
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr
        ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }

  tiler_rows = (guint) sqrt (num_sources);
  tiler_columns = (guint) ceil (1.0 * num_sources / tiler_rows);
  /* we set the tiler properties here */
  g_object_set (G_OBJECT (tiler), "rows", tiler_rows, "columns", tiler_columns,
      "width", TILED_OUTPUT_WIDTH, "height", TILED_OUTPUT_HEIGHT, NULL);

  g_object_set (G_OBJECT (nvosd), "process-mode", OSD_PROCESS_MODE,
      "display-text", OSD_DISPLAY_TEXT, NULL);

  g_object_set (G_OBJECT (sink), "qos", 0, NULL);

  /* Set all the necessary properties of the infer plugin element,
   * Enable Output tensor meta, we can probe PGIE and
   * SGIEs buffer to parse tensor output data of models */
  if (!is_nvinfer_server) {
    /* nvinfer Output tensor meta can be enabled by set "output-tensor-meta=true"
     * here or enable this attribute in config file. */
    g_object_set (G_OBJECT (pgie), "config-file-path", INFER_PGIE_CONFIG_FILE,
        "output-tensor-meta", TRUE, "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", INFER_SGIE1_CONFIG_FILE,
        "output-tensor-meta", TRUE, "process-mode", 2, NULL);
    //g_object_set (G_OBJECT (sgie2), "config-file-path", INFER_SGIE2_CONFIG_FILE,
        //"output-tensor-meta", TRUE, "process-mode", 2, NULL);
    //g_object_set (G_OBJECT (sgie3), "config-file-path", INFER_SGIE3_CONFIG_FILE,
        //"output-tensor-meta", TRUE, "process-mode", 2, NULL);
  } else {
    /* nvinferserver output tensor meta can be enabled in config file.
     * see output_control { output_tensor_meta: true }*/
    g_object_set (G_OBJECT (pgie), "config-file-path", INFERSERVER_PGIE_CONFIG_FILE,
        "batch-size", num_sources, NULL);
    g_object_set (G_OBJECT (sgie1), "config-file-path", INFERSERVER_SGIE1_CONFIG_FILE,
        "process-mode", 2, NULL);
    //g_object_set (G_OBJECT (sgie2), "config-file-path", INFERSERVER_SGIE2_CONFIG_FILE,
        //"process-mode", 2, NULL);
    //g_object_set (G_OBJECT (sgie3), "config-file-path", INFERSERVER_SGIE3_CONFIG_FILE,
        //"process-mode", 2, NULL);
  }
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* decoder | pgie1 | sgie1 | sgie2 | sgie3 | etc.. */
#ifdef PLATFORM_TEGRA
  gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, sgie1, queue3, tiler, queue4,
      nvvidconv, queue5, nvosd, queue6, transform, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, sgie1,queue3, tiler, queue4,
        nvvidconv, queue5, nvosd, queue6, transform, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#else
gst_bin_add_many (GST_BIN (pipeline), queue1, pgie, queue2, sgie1, queue3, tiler, queue4,
    nvvidconv, queue5, nvosd, queue6, sink, NULL);
  /* we link the elements together
   * nvstreammux -> nvinfer -> nvtiler -> nvvidconv -> nvosd -> video-renderer */
  if (!gst_element_link_many (streammux, queue1, pgie, queue2, sgie1, queue3, tiler, queue4,
        nvvidconv, queue5, nvosd, queue6, sink, NULL)) {
    g_printerr ("Elements could not be linked. Exiting.\n");
    return -1;
  }
#endif

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  osd_sink_pad = gst_element_get_static_pad (nvosd, "sink");
  if (!osd_sink_pad)
    g_print ("Unable to get sink pad\n");
  else
    gst_pad_add_probe (osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
        osd_sink_pad_buffer_probe, NULL, NULL);
  gst_object_unref (osd_sink_pad);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the source pad of PGIE's next queue element, since by that time, PGIE's
   * buffer would have had got tensor metadata. */
  queue_src_pad = gst_element_get_static_pad (queue2, "src");
  gst_pad_add_probe (queue_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
      pgie_pad_buffer_probe, NULL, NULL);

  /* Add probe to get informed of the meta data generated, we add probe to
   * the sink pad of tiler element which is just after all SGIE elements.
   * Since by that time, GstBuffer would have had got all SGIEs tensor
   * metadata. */
  tiler_sink_pad = gst_element_get_static_pad (tiler, "sink");
  gst_pad_add_probe (tiler_sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
      sgie_pad_buffer_probe, NULL, NULL);

  /* Set the pipeline to "playing" state */
  g_print ("Now playing...\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  

  /* Iterate */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
