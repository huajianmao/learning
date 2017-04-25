import React, { Component } from 'react';

import { StyleSheet, View, WebView, InteractionManager } from 'react-native';
import { Actions } from 'react-native-router-flux';

class WebScene extends Component {
  state: {
    source: Object
  }

  constructor(props: Object) {
    super(props)
    this.state = {
      source: {}
    }
  }

  componentDidMount() {
    InteractionManager.runAfterInteractions(() => {
      this.setState({source: {url: this.props.url}})
    })
  }

  render() {
    return (
      <View style={styles.container}>
        <WebView ref='webview'
          automaticallyAdjustContentInsets={false}
          style={styles.webview}
          source={this.state.source}
          onLoadEnd={(e) => this.onLoadEnd(e)}
          scalesPageToFit={true} />
      </View>
    )
  }

  onLoadEnd(e: any) {
    Actions.refresh({ title: e.nativeEvent.title })
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#2c3e50',
  },
  webview: {
    flex: 1,
    backgroundColor: 'white',
  }
});

export default WebScene;
