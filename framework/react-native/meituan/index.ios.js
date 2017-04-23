/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 * @flow
 */

import React, { Component } from 'react';

import RootScene from './src/RootScene';

import { AppRegistry } from 'react-native';

export default class meituan extends Component {
  render() {
    return (
      <RootScene />
    );
  }
}

AppRegistry.registerComponent('meituan', () => meituan);

