# 데모 검수 가이드

## 공통 안내

세 HTML은 모두 self-contained 파일이다. GitHub에서 클릭하면 HTML 문법이
보일 수 있으므로 저장소를 clone하거나 파일을 다운로드한 뒤 브라우저로
연다.

이 데모들의 목적은 최종 결과를 주장하는 것이 아니라 다음 질문을 순서대로
검증하는 것이다.

1. 장면 자체가 연구하려는 오프더볼 상호작용을 포함하는가?
2. 어떤 수비수가 runner와 다른 공격 옵션의 균형을 바꿀 수 있는가?
3. 현재 geometry screen이 그 관계를 상식적으로 표현하는가?

## 1. Clear-core scene audit

파일:
[`clear_core_scene_audit.html`](../examples/research_audit/current_demos/clear_core_scene_audit.html)

이 페이지는 second-pass에서 포함으로 확정된 장면 8개를 실제 움직임으로만
보여준다. 보류된 2개 장면은 공유용 데모와 이후 개발 payload에서 제외했다.

- 금색 테두리와 궤적: focal runner
- 흰 테두리: `t=0`의 ball carrier
- 빨간색: 공격팀
- 파란색: 수비팀
- `t=0`: 검출된 run onset

확인할 것은 다음이다.

- `t=0`이 새로운 움직임의 시작으로 납득되는가?
- runner에게 반응할 수 있는 수비수가 실제로 보이는가?
- 그 수비수가 runner를 우선하면 ball carrier나 다른 동료의 공간이 열리는가?
- 다른 옵션을 우선하면 runner가 직접 위협을 유지하는가?

이 페이지에는 가상 수비 궤적이나 위협 점수가 없다. 따라서 “모델이
dilemma를 찾았다”가 아니라 “모델링할 구조가 눈으로 보인다”만 판단한다.

## 2. Meeting scene gallery

파일:
[`meeting_scene_gallery.html`](../examples/research_audit/current_demos/meeting_scene_gallery.html)

여기에는 최종 포함된 고유 observed 장면 8개가 있다. 각 장면에서 먼저 실제
움직임을 본다.

- runner와 ball carrier의 역할이 자연스러운가?
- 사람이 기록한 주 반응 수비수가 타당한가?
- 사람이 기록한 파생 옵션이 runner의 움직임과 연결되는가?
- 특정 선수를 beneficiary로 미리 고정하지 않아야 하는 장면인가?

Klaus 장면만 `실제 움직임`과 `Dynamic prototype`을 전환할 수 있다. Dynamic
prototype은 과거 구현을 보존한 mechanism demo이며 Kownacki carry를 파생
옵션으로 고정한다. 최종 beneficiary 선정 또는 현재 최적 수비 결과가 아니다.

## 3. Structural local-game audit

파일:
[`structural_local_game_audit.html`](../examples/research_audit/current_demos/structural_local_game_audit.html)

이 페이지는 완성된 결과가 아니라, 지금까지 구현한 국소적 게임 구성의
working prototype이다. 현재 표현이 축구적으로 타당한지 의견을 받기 위해
공유하며, 숫자나 수비 경로를 최종 결과로 해석하지 않는다.

각 장면은 후보 수비수 3명과 공격 옵션 5명을 비교한다. 공격 옵션에는 ball
carrier가 항상 포함된다.

각 cell의 핵심 표시는 다음과 같다.

- `O`: 수비수가 runner에 대응할 때 해당 option 통제가 얼마나 나빠지는가?
- `R`: 수비수가 해당 option을 통제할 때 runner 통제가 얼마나 나빠지는가?
- structural score: `O`와 `R` 중 작은 값

따라서 두 값이 모두 클 때만 양방향 allocation conflict가 있다는 구조적
신호가 된다.

검수할 때는 다음 순서가 좋다.

1. 선택된 세 수비수 중 실제로 상호작용에 관련 있는 선수가 포함됐는가?
2. ball carrier와 육안으로 중요한 동료 옵션이 다섯 열 안에 포함됐는가?
3. runner 억제 경로가 과거 위치가 아니라 움직이는 goal-side 공간을 막는가?
4. option 억제 경로가 해당 공격 옵션의 미래 진행 공간을 막는가?
5. `O`와 `R`이 큰 cell이 실제 장면에서도 trade-off로 보이는가?

숫자의 의미를 과대해석하면 안 된다. 현재 공격수 궤적은 관측된 미래이며,
수비 궤적은 bounded geometry response다. `O`, `R`, structural score는 패스
성공 확률, xG, OBSO 또는 최종 threat가 아니다.

## 피드백 우선순위

찬의·준현씨에게는 구현 세부보다 다음 의견을 먼저 받는 것이 좋다.

1. defender-dependent local option set이 축구적으로 설득력 있는가?
2. 선수와 단순 거리가 아니라 moving goal-side control을 보는 것이 적절한가?
3. direct threat와 derived threat 외에 빠진 핵심 축이 있는가?
4. 현재 structural screen 이후 어떤 value component를 먼저 검증해야 하는가?
5. 어느 수준의 양방향 cross-cost를 실제 dilemma라고 부를 수 있는가?
