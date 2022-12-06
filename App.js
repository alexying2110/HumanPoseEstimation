import React, { useState, useEffect } from 'react';
import { StatusBar } from 'expo-status-bar';
import { Button, Image, StyleSheet, Text, View } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as tf from '@tensorflow/tfjs';
import * as posedetection from '@tensorflow-models/pose-detection';
import {
  bundleResourceIO,
  decodeJpeg,
} from '@tensorflow/tfjs-react-native';
import { ExpoWebGLRenderingContext } from 'expo-gl';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import * as FileSystem from 'expo-file-system';
import Svg, { Circle, Line } from 'react-native-svg';
import { manipulateAsync } from 'expo-image-manipulator';

export default class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      isTfReady: false,
      image: "file:///data/user/0/host.exp.exponent/cache/ExperienceData/%2540smartest%252Fhpe/ImageManipulator/33a07f8b-f371-416d-85e3-bd44f8ca889c.jpg",
      time: null,
      poses: null,
    };
  }

  async componentDidMount() {
    try {
      await tf.ready();
      this.setState({
        isTfReady: true,
      });
    } catch (e) {
      console.log(e);
    }

    try {
      this.model = await posedetection.createDetector(
        posedetection.SupportedModels.MoveNet,
        {
          modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          enableSmoothing: true,
        }
      );
    } catch (e) {
      console.log(e)
    }
  }


  pickImage = async () => {
    let image;
    try {
      let result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [2, 3],
        quality: 1,
      });

      image = await manipulateAsync(
        result.assets[0].uri,
        [{ resize: {height: 600, width: 400} }],
      );
    } catch (e) {
      console.log(e);
    }

    this.setState({
      image: image.uri
    })
  };

  interp = async () => {
    console.log(this.state.image);
    try {
      const imgB64 = await FileSystem.readAsStringAsync(this.state.image, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const imgBuffer = tf.util.encodeString(imgB64, 'base64').buffer;
      const raw = new Uint8Array(imgBuffer)  
      const imageTensor = decodeJpeg(raw);
      let t1 = Date.now();
      let poses = await this.model.estimatePoses(imageTensor);
      let t2 = Date.now();
      this.setState({
        poses,
        time: t2 - t1
      });
    } catch (e) {
      console.log(e);
    }
  }

  renderSVG = () => {

    const connections = {
      left_shoulder: [
        "right_shoulder",
        "left_hip",
        "left_elbow",
      ],
      left_elbow: [
        "left_wrist"
      ],
      right_shoulder: [
        "right_elbow",
        "right_hip",
      ],
      right_elbow: [
        "right_wrist"
      ],
      left_hip: [
        "left_knee",
        "right_hip"
      ],
      right_hip: [
        "right_knee"
      ],
      left_knee: [
        "left_ankle"
      ],
      right_knee: [
        "right_ankle"
      ],
    }

    let poses = this.state.poses;
    const MIN_KEYPOINT_SCORE = 0.5;
    if (poses != null && poses.length > 0) {
      const keypoints = poses[0].keypoints
        .filter(k => k.score > MIN_KEYPOINT_SCORE)
        .map(k => {
          console.log('here', k);
          const x = k.x;
          const y = k.y;
          return (
            <Circle
              key={`skeletonkp_${k.name}`}
              cx={x}
              cy={y}
              r='4'
              strokeWidth='2'
              fill='#0000AA'
              stroke='white'
            />
          );
        });

      let kpObj = poses[0].keypoints
        .filter(k => k.score > MIN_KEYPOINT_SCORE)
        .reduce((acc, curr) => {
          acc[curr.name] = curr;
          return acc;
        }, {})

      Object.entries(connections).forEach(([key, val]) => {
        if (kpObj[key]) {
          val.forEach(conn => {
            if (kpObj[conn]) {
              keypoints.push(
                <Line
                  x1={kpObj[key].x}
                  y1={kpObj[key].y}
                  x2={kpObj[conn].x}
                  y2={kpObj[conn].y}
                  stroke={"#0000AA"}
                  strokeWidth={3}
                />
              );
            }
          })
        }

      })
      console.log(keypoints);

      return keypoints;
    } else {
      return <View/>;
    }
  }

  render() {
    return (
      <GestureHandlerRootView style={{ flex: 1 }}>
        <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
          <Button title="Pick an image from camera roll" onPress={this.pickImage} />
          <Button title="Estimate pose" onPress={this.interp} />
          <View>
            {this.state.image && <Image source={{ uri: this.state.image }} style={{ width: 400, height: 600 }} />}
            <Svg 
              height={"600"}
              width={"400"}
              style={{
                borderWidth: 1,
                width: '100%',
                height: '100%',
                position: 'absolute',
                zIndex: 30,
              }}
            >
              {this.renderSVG()}
            </Svg>
          </View>
          {this.state.time && <Text>Model eval time: {this.state.time} ms</Text>}
        </View>
      </GestureHandlerRootView>
    );
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
