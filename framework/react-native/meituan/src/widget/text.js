import React from 'react'

import { Text, StyleSheet } from 'react-native'

export function Heading2({style, ...props}: Object): ReactElement {
  return <Text style={[style.h2, style]} {...props} />
}

export function Paragraph({style, ...props}: Object): ReactElement {
  return <Text style={[styles.p, style]} {...props} />
}

const styles = StyleSheet.create({
  h2: {
    fontSize: 14,
    color: '#222222'
  },
  p: {
    fontSize: 13,
    color: '#777777',
  }
})