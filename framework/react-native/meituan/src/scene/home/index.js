import React, { Component } from 'react';

import { StyleSheet, Text, View, TouchableOpacity, Image, ListView } from 'react-native';

import NavigationItem from '../../widget/navigationitem'

import color from '../../widget/color'
import screen from '../../common/screen'
import system from '../../common/system'
import { Paragraph } from '../../widget/text'

class HomeScene extends Component {
  static renderRightButton = () => {
    return (
      <NavigationItem icon={require('../../img/home/icon_navigationItem_message_white@2x.png')}
                      onPress={() => {}}/>
    )
  }

  static renderTitle = () => {
    return (
      <TouchableOpacity>
        <Image source={require('../../img/home/search_icon.png')} style={styles.searchIcon} />
        <Paragraph>一点点</Paragraph>
      </TouchableOpacity>
    )
  }

  static renderLeftButton = () => {
    return (
      <NavigationItem title="北京" titleStyle={{color: 'white'}} onPress={() => {}} />
    )
  }

  state: {
    discounts: Array<Object>,
    dataSource: ListView.DataSource
  }

  render() {
    return (
      <View>
        <Text>
          HomeScene
        </Text>
      </View>
    )
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: color.background
  },
  recommendHeader: {
    height: 35,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: color.border,
    paddingVertical: 8,
    paddingLeft: 20,
    backgroundColor: 'white'
  },
  searchBar: {
    width: screen.width * 0.7,
    height: 30,
    borderRadius: 19,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'white',
    alignSelf: 'center',
    marginTop: system.isIOS ? 25 : 13,
  },
  searchIcon: {
    width: 20,
    height: 20,
    margin: 5,
  }
})

export default HomeScene;