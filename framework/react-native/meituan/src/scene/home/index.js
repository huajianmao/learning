import React, { Component } from 'react';

import { StyleSheet, Text, View, TouchableOpacity, Image, ListView } from 'react-native';

import RefreshListView, { RefreshState } from '../../widget/refreshlistview'
import NavigationItem from '../../widget/navigationitem'

import color from '../../widget/color'
import screen from '../../common/screen'
import system from '../../common/system'
import { Heading2, Paragraph } from '../../widget/text'

import api from '../../api/index'

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

  constructor(props: Object) {
    super(props)
    let ds = new ListView.DataSource({rowHasChanged: (r1, r2) => r1 !== r2})
    this.state = {
      discounts: [],
      dataSource: ds.cloneWithRows([])
    }
  }

  requestData() {
    this.requestDiscount()
    this.requestRecommend()
  }

  requestDiscount() {
    fetch(api.discount)
      .then((response) => response.json())
      .then((json) => {
        this.setState({...this.state, discounts: json.data})
      })
      .catch((error) => {alert(error)})
  }

  requestRecommend() {
    fetch(api.recommend)
      .then((response) => response.json())
      .then((json) => {
        let datalist = json.data.map((info) => {
          return {

          }
        })

        this.setState({dataSource: this.state.dataSource.cloneWithRows(datalist)})
        setTimeout(() => {
          this.refs.listview.endRefreshing(RefreshState.NoMoreData)
        }, 500)
      })
      .catch((error) => {
        this.refs.listview.endRefreshing()
      })
  }

  componentDidMount() {
    this.refs.listview.startHeaderRefreshing(RefreshState.Failure)
  }

  render() {
    return (
      <View>
        <RefreshListView ref="listview"/>
      </View>
    )
  }

  renderHeader() {
    return (
      <View>

        <View style={styles.recommendHeader}>
          <Heading2>猜你喜欢</Heading2>
        </View>
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