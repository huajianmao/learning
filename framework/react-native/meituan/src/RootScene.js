
import React, { Component } from 'react';
import { StyleSheet, Text, View, StatusBar } from 'react-native';
import { Router, Scene, Actions } from 'react-native-router-flux';

import HomeScene from './scene/home/index'
import NearbyScene from './scene/nearby/index'
import DiscoverScene from './scene/discover/index'
import OrderScene from './scene/order/index'
import MineScene from './scene/mine/index'
import PurchaseScene from './scene/purchase/index'
import WebScene from './scene/web/index'

import color from './widget/color'
import screen from './widget/screen'
import system from './widget/system'

import TabItem from './widget/tabitem'

class RootScene extends Component {
  render() {
    return (
      <Router ref='router'
        titleStyle={styles.navigationBarTitle}
        barButtonIconStyle={styles.navigationBarButtonIcon}
        navigationBarStyle={styles.navigationBarStyle}
        getSceneStyle={this.sceneStyle}
        panHandlers={null}
        animationStyle={this.animate}
        onSelect={this.onSelect}
        onBack={this.onBack}>
        <Scene initial key='tabBar' tabs
          tabBarStyle={styles.tabBar}
          tabBarSelectedItemStyle={styles.tabBarSelectedItem}
          tabBarSelectedTitleStyle={styles.tabBarSelectedTitle}
          tabBarUnselectedTitleStyle={styles.tabBarUnselectedTitle}>
          <Scene key="home" component={HomeScene} title="首页" titleStyle={{ color: 'white' }}
                 navigationBarStyle={{ backgroundColor: color.theme }}
                 statusBarStyle='light-content'
                 image={require('./img/tabbar/pfb_tabbar_homepage@2x.png')}
                 selectedImage={require('./img/tabbar/pfb_tabbar_homepage_selected@2x.png')}
                 icon={TabItem} />
          <Scene key="nearby" component={NearbyScene} title="附近"
                 image={require('./img/tabbar/pfb_tabbar_merchant@2x.png')}
                 selectedImage={require('./img/tabbar/pfb_tabbar_merchant_selected@2x.png')}
                 icon={TabItem} />
          <Scene key="discover" component={DiscoverScene} title="逛一逛"
                 image={require('./img/tabbar/pfb_tabbar_discover@2x.png')}
                 selectedImage={require('./img/tabbar/pfb_tabbar_discover_selected@2x.png')}
                 icon={TabItem} />
          <Scene key="order" component={OrderScene} title="订单"
                 image={require('./img/tabbar/pfb_tabbar_order@2x.png')}
                 selectedImage={require('./img/tabbar/pfb_tabbar_order_selected@2x.png')}
                 icon={TabItem} />
          <Scene key="mine" component={MineScene} title="我的" hideNavBar
                 statusBarStyle='light-content'
                 image={require('./img/tabbar/pfb_tabbar_mine@2x.png')}
                 selectedImage={require('./img/tabbar/pfb_tabbar_mine_selected@2x.png')}
                 icon={TabItem} />
        </Scene>
        <Scene key='web' component={WebScene} title='加载中' hideTabBar clone />
        <Scene key='purchase' component={PurchaseScene} title='团购详情' hideTabBar clone />
      </Router>
    )
  }

  sceneStyle = (props: Object, computedProps: Object) => {
    const style = {
      flex: 1,
      backgroundColor: color.theme,
      shadowColor: null,
      shadowOffset: null,
      shadowOpacity: null,
      shadowRadius: null,
      marginTop: 0,
      marginBottom: 0,
    };

    if (computedProps.isActive) {
      style.marginTop = computedProps.hideNavBar ? (system.isIOS ? 20 : 0) : (system.isIOS ? 64 : 54);
      style.marginBottom = computedProps.hideTabBar ? 0 : 50;
    }
    return style;
  }

  animate = props => {
    const { position, scene } = props;

    const index = scene.index;
    const inputRange = [index - 1, index + 1];
    const outputRange = [screen.width, -screen.width];

    const translateX = position.interpolate({ inputRange, outputRange });
    return { transform: [{ translateX }] };
  }

  onSelect = (el: Object) => {
    const { sceneKey, statusBarStyle } = el.props
    if (statusBarStyle) {
      StatusBar.setBarStyle(statusBarStyle, false)
    } else {
      StatusBar.setBarStyle('default', false)
    }
    Actions[sceneKey]()
  }
  onBack = (el: Object) => {
    if (el.sceneKey == 'home' && el.children.length == 2) {
      StatusBar.setBarStyle('light-content', false)
    }
    Actions.pop()
  }
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#ffffff',
  },
  tabBarSelectedItem: {
    backgroundColor: '#ffffff',
  },

  tabBarSelectedTitle: {
    color: color.theme,
  },
  tabBarUnselectedTitle: {
    color: '#979797',
  },

  tabBarSelectedImage: {
    tintColor: color.theme,
  },
  tabBarUnselectedImage: {
    tintColor: '#979797'
  },

  navigationBarStyle: {
    backgroundColor: 'white'
  },
  navigationBarTitle: {
    color: '#333333'
  },
  navigationBarButtonIcon: {
    tintColor: color.theme
  },
});

export default RootScene;
